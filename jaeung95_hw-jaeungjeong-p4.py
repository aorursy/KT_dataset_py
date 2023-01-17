

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import cv2 

import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score

from os.path import join, basename

from os import listdir

from PIL import Image 

from PIL import ImageFilter

from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import accuracy_score

dataset_train = "../input/2019-fall-pr-project/train/train/"

print(listdir(dataset_train))

train_imgs = [join(dataset_train,f) for f in listdir(dataset_train)] #상대경로



resize=[]



for i in train_imgs:

  img = cv2.imread(i)

  img1=cv2.resize(img,dsize=(32,32))

  b = np.reshape(img1, (1,np.product(img1.shape)))

  resize.append(b[0,:])



labels=[] 

print(os.listdir(dataset_train))

for i in os.listdir(dataset_train):

  if 'dog' in i:

    labels.append('1')

  else:

    labels.append('0')

resize1, subtestImage, label1, subtestLabel = train_test_split(resize, labels, test_size=0.25, random_state=42)
clf=LinearSVC(C=1E5,class_weight='balanced')

clf.fit(subtestImage[0:4000], subtestLabel[0:4000])
label3=clf.predict(resize1[0:500])

accuracy_score(label1[0:500],label3)


dataset_test = "../input/2019-fall-pr-project/test1/test1/"

test_imgs = [join(dataset_test,f) for f in listdir(dataset_test)]

test_size=[]

for i in test_imgs:

  img = cv2.imread(i)

  img1=cv2.resize(img,dsize=(32,32)) #32,32 사이즈로 변경

  b = np.reshape(img1, (1,np.product(img1.shape))) #1d vector로 변경

  test_size.append(b[0,:])
result=clf.predict(test_size)
import pandas as pd

s = pd.Series(result)

data={'id':range(1,5001),'label':s}

df = pd.DataFrame(data)

df.index += 1 

df.to_csv('results-yk-v2.csv',index=True,index_label='id', header=True,columns=["label"])