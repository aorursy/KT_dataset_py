# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import cv2
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image=cv2.imread(x)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.3,5)
if faces is ():
    print('not_found')
for (x,y,w,h) in faces:
    cv2.ractangle(images,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('facedetecion',image)
    cv2.waitkey(0)
    
cv2.destroyAllWindows()
    