import math

import numpy as np
import cv2
import HandTrackingModule as htm

import mediapipe as mp
import time
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
# cTime = 0
detector = htm.HandDetector(detectionCon=0.75, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
# print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(-20.0, None)
vol = 0
volBar = 400
area = 0
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(area)
        if 100 < area < 1000:
            # print(lmList[8])
            # print(bbox)
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # convrt volume

            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            # print(volPer)
            # volume.SetMasterVolumeLevel(vol, None)
            smooth = 10
            volPer = smooth * round(volPer / smooth)

            fingers = detector.fingersUp()
            print("^___^:", fingers)
            if fingers[4]==False:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)

                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

        # Drawing
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40,450),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
