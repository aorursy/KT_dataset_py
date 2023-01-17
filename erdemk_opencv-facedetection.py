# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import cv2
import numpy as np

import cv2 as cv



%matplotlib inline

from matplotlib import pyplot as plt



face_cascade = cv.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv.CascadeClassifier('../input/haarcascades/haarcascade_eye.xml')



#img = cv.imread('../input/burak-can-yuz/burak_can.jpg')

img = cv.imread('../input/test01/test01.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



plt.imshow(img)

plt.show()

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:

    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    roi_gray = gray[y:y+h, x:x+w]

    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:

        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



plt.imshow(img)

plt.show()
