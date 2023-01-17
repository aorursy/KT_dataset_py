import cv2
import numpy as np

cap = cv2.VideoCapture('Video/face_track.mp4')

ret , frame = cap.read()

face_casc = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_casc.detectMultiScale(frame)

face_x,face_y,w,h=tuple(face_rects[0])
track_window=(face_x,face_y,w,h)

roi = frame[face_y:face_y+h,
           face_x:face_x+w]
hsv_roi = cv2.cvtColor(roi,
                      cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi],
                       [0],
                       None,
                       [180],
                       [0,180])

cv2.normalize(roi_hist,
              roi_hist,
             0,
             255,
             cv2.NORM_MINMAX);

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
while True:

    # capture video
    ret,frame = cap.read()
    
    # if statement
    if ret == True:
    
        # Frame in HSV
        hsv = cv2.cvtColor(frame,
                          cv2.COLOR_BGR2HSV)
        
        # Calculate the base of ROI
        dest_roi = cv2.calcBackProject([hsv],
                                      [0],
                                      roi_hist,
                                      [0,180],
                                      1)
        
        # Meanshift to get the new coordinates of rectangle
        ret , track_window = cv2.meanShift(dest_roi,
                                          track_window,
                                          term_crit)
        
        # Draw new rectangle on image
        x,y,w,h = track_window
        
        # Open new window and display
        img2 = cv2.rectangle(frame, (x,y),
                            (x+w,y+h),
                            (255,255,0),
                            3)
        cv2.imshow('FaceTracker', img2)
        
        # Close window
        if cv2.waitKey(50) & 0xFF == 27:
            break
        
    # else statement
    else :
        break
    
# Release and Destroy
cap.release()
cv2.destroyAllWindows()