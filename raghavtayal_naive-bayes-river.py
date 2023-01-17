import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#from skimage.io import imread

from PIL import Image

import cv2

import math

import string

import os

print(os.listdir("../input"))
img1=cv2.imread('../input/1.png',0)

img2=cv2.imread('../input/2.png',0)

img3=cv2.imread('../input/3.png',0)

img4=cv2.imread('../input/4.png',0)

#img1 = img1.astype(float)

#img2 = img2.astype(float)

#img3 = img3.astype(float)

#img4 = img4.astype(float)
mean = np.zeros((4,1))

cnt = 0

df_river = pd.DataFrame(columns=[0, 1, 2, 3])

for i in range(159,169) :

    for j in range(155,160):

        mean[0][0] += img1[i][j]

        mean[1][0] += img2[i][j]

        mean[2][0] += img3[i][j]

        mean[3][0] += img4[i][j]

        df_river.loc[cnt] = [img1[i][j],img2[i][j],img3[i][j],img4[i][j]]

        cnt+=1

river = np.array([mean[0][0]/50,mean[1][0]/50,mean[2][0]/50,mean[3][0]/50])

river = river[:,np.newaxis]
mean = np.zeros((4,1))

cnt = 0

df_nonriver = pd.DataFrame(columns=[0, 1, 2, 3])

for i in range(327,337) :

    for j in range(220,230):

        mean[0][0] += img1[i][j]

        mean[1][0] += img2[i][j]

        mean[2][0] += img3[i][j]

        mean[3][0] += img4[i][j]

        df_nonriver.loc[cnt] = [img1[i][j],img2[i][j],img3[i][j],img4[i][j]]

        cnt+=1

nonriver = np.array([mean[0][0]/100,mean[1][0]/100,mean[2][0]/100,mean[3][0]/100])

nonriver = nonriver[:,np.newaxis]
train_river = np.array(df_river)

Rivercovar = np.cov(train_river.astype(float),rowvar=0)

print(Rivercovar)
train_nonriver = np.array(df_nonriver)

NonRivercovar = np.cov(train_river.astype(float),rowvar=0)
p1 = np.ones((512,512))

det_covar_river = np.linalg.det(Rivercovar)

RiverClass = np.zeros((512,512))

for i in range(0,512):

    for j in range(0,512):

        InvCovar = np.linalg.inv(Rivercovar)

        temp = np.array([np.subtract(img1[i][j],river[0]), np.subtract(img2[i][j],river[1]), np.subtract(img3[i][j],river[2]), np.subtract(img4[i][j],river[3])])

        #print(np.transpose(temp).shape)

        RiverClass[i][j] += np.matmul(np.matmul(np.transpose(temp),InvCovar),temp)

        p1[i][j] *= (1/math.sqrt(det_covar_river))*math.exp(-0.5 * RiverClass[i][j])

print(RiverClass)
p2 = np.ones((512,512))

det_covar_nonriver = np.linalg.det(NonRivercovar)

NonRiverClass = np.zeros((512,512))

for i in range(0,512):

    for j in range(0,512):

        InvCovar = np.linalg.inv(NonRivercovar)

        temp = np.array([np.subtract(img1[i][j],nonriver[0]), np.subtract(img2[i][j],nonriver[1]), np.subtract(img3[i][j],nonriver[2]), np.subtract(img4[i][j],nonriver[3])])

        #print(np.transpose(temp).shape)

        NonRiverClass[i][j] += np.matmul(np.matmul(np.transpose(temp),InvCovar),temp)

        p2[i][j] *= (1/math.sqrt(det_covar_nonriver))*math.exp(-0.5*NonRiverClass[i][j]) 
result = np.zeros((512,512))

P1 = 0.7

P2 = 0.3

for i in range(512):

    for j in range(512):

        p1[i][j] *= P1

        p2[i][j] *= P2

        if(p1[i][j] >= p2[i][j]) :

            result[i][j] += 255

        else :

            result[i][j] += 0

Image.fromarray(np.uint8(result))
