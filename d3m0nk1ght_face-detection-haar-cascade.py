# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2

import matplotlib.pyplot as plt

face_cascade=cv2.CascadeClassifier("../input/cascade-face-default/haarcascade_frontalface_default.xml")

left_eye_cascade=cv2.CascadeClassifier("../input/left-eye/haarcascade_lefteye_2splits.xml")

right_eye_cascade=cv2.CascadeClassifier("../input/right-eye/haarcascade_righteye_2splits.xml")

img=cv2.imread("../input/testjaye/mj.jpg")

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3, 5)

print(type(faces))

print(faces)

# Any results you write to the current directory are saved as output.

for (x,y,w,h) in faces:

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    roi=img[y:y+h,x:x+w]

    roi_gray=gray[y:y+h,x:x+w]

    left_eyes=left_eye_cascade.detectMultiScale(roi_gray)

    right_eyes=right_eye_cascade.detectMultiScale(roi_gray)

    for (lex,ley,lew,leh) in left_eyes:

        cv2.rectangle(roi,(lex,ley),(lex+lew,ley+leh),(255,0,0),2)

    for (rex,rey,rew,reh) in right_eyes:

        cv2.rectangle(roi,(rex,rey),(rex+rew,rey+reh),(255,0,0),2)

plt.title('Face detcted in original image')

plt.imshow(img,cmap='gray')



plt.show()

    