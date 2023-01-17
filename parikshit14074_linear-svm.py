# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.decomposition import PCA

import os

import cv2

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train_file_name_class= pd.read_csv("../input/train.csv", header=0)

# print(train_file_name_class.head())

train_file_names=train_file_name_class['id']

trainY=train_file_name_class['has_cactus']

trainX=[]

for file_name in train_file_names.values:

  img=cv2.imread('../input/train/train/'+file_name)

  img = img.reshape(32*32*3,)

  img=img/255

  trainX.append(img)

print(len(trainX))

  
test_file_name_class= pd.read_csv("../input/sample_submission.csv", header=0)

# print(test_file_name_class.head())

test_file_names=test_file_name_class['id']

testX=[]

for file_name in test_file_names.values:

  img=cv2.imread('../input/test/test/'+file_name)

  #print(img.shape)

  img = img.reshape(32*32*3,)

  img=img/255

  testX.append(img)

print(len(testX))
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler= scaler.fit(trainX+testX)

trainX = scaler.transform(trainX)

testX = scaler.transform(testX)
from sklearn.svm import LinearSVC



clf = LinearSVC(dual=False)

clf=clf.fit(trainX, trainY.values)

testY=clf.predict(testX)

test_file_name_class['has_cactus']=testY

test_file_name_class.to_csv('sample_submission.csv', index=False)