import cv2
import time
import numpy as np

cam = cv2.VideoCapture(0)

for i in range(60):
    ret,background = cam.read()

background = cv2.flip(background,1)

while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)
    mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background,background,mask=mask1)
    res2 = cv2.bitwise_and(frame,frame,mask=mask2)
    final_output = cv2.addWeighted(res1,1,res2,1,0)

    cv2.imshow('Invisibility Cloak',final_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
