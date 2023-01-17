# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import copy

import cv2

import matplotlib.pyplot as plt

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
DATA_PATH = "../input"

TRAIN_PATH = os.path.join(DATA_PATH, 'Train')



for i in range(6):

    img = cv2.imread(TRAIN_PATH+'/'+str(i)+'.jpg')

    



    img = cv2.resize(img, (512,512))

    img2 = copy.deepcopy(img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(img_hsv)



    lower_green = np.array([0,125,0])

    upper_green = np.array([255,200,255])

    img4 = cv2.inRange(img2,lower_green,upper_green)



    no_seals = np.where(v < 150)

    img2[no_seals] = 0



    others = np.where(s < 40)



    img2[others] = 0

    

    img3 = copy.deepcopy(img2)

    lower_blue = np.array([102,127,150])

    upper_blue = np.array([163,185,201])

    img3 = cv2.inRange(img3,lower_blue,upper_blue)



    blue_values = np.where(img3==0)

    img2[blue_values] = 0

    

# ----------------------------------------------------------------------------#

    

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img3 = copy.deepcopy(img)



    img4 = copy.deepcopy(img2)

    img4[img4 > 0]=255



    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv2.dilate(img4,kernel,iterations=2)





    _,contours, _= cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt,True)

        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),2)



    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    #res = np.hstack((img,img2)) 

    

    f, ax = plt.subplots(1, 2, figsize=(12,8))

    (ax1, ax2) = ax.flatten()



    ax1.imshow(img)

    ax1.set_title("Original")

    ax2.imshow(img2)

    ax2.set_title("Segmented")

    plt.show()

     

    f, ax = plt.subplots(figsize=(8,8))   

    ax.imshow(img3)

    ax.set_title("Proposed")

    plt.show()