# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2



reeses=cv2.imread('../input/reeses_puffs.png',0)

cereals=cv2.imread('../input/many_cereals.jpg',0)



def display(img):

    # in order to modify the size

    fig = plt.figure(figsize=(12,8))

    # adding multiple Axes objects 

    ax = fig.add_subplot(111)

    ax.imshow(img,cmap='gray')



display(reeses)

display(cereals)



orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(reeses,mask=None)

kp2,des2=orb.detectAndCompute(cereals,mask=None)    



bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)



matches=bf.match(des1,des2)



for match in matches:

    print(match.distance)

    

matches=sorted(matches,key=lambda x:x.distance)  

reese_matches=cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:20],None,flags=2)

display(reese_matches)

# Any results you write to the current directory are saved as output.









##Brief algorithm

# Initiate STAR detector



star = cv2.xfeatures2d.StarDetector_create()

    

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()



# find the keypoints with STAR

kp1 = star.detect(reeses,None)



# compute the descriptors with BRIEF

kp1, des1 = brief.compute(reeses,kp)



#print brief.getInt('bytes')

kp2=star.detect(cereals,mask=None)   

kp2, des2 = brief.compute(cereals,kp2)

#kp1,des1=sift.detectAndCompute(reeses,None)

#kp2,des2=sift.detectAndCompute(cereals,None)



bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)



matches=bf.match(des1,des2)





matches=sorted(matches,key=lambda x:x.distance)  

reese_matches=cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:20],None,flags=2)

display(reese_matches)
