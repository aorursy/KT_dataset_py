# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

from tqdm import tqdm

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_csv = pd.read_csv("../input/train.csv")
# x = np.zeros((30592, 224,224,3), dtype=np.int32)

train_list = []
# cntr = 0

for i in data_csv.id:

    temp = np.array(Image.open('../input/train/train/'+i))

    reflect = cv2.copyMakeBorder(temp,96,96,96,96,cv2.BORDER_REFLECT)

    train_list.append(reflect)

#     cntr = cntr+1
train_list = np.array(train_list)

train_list = np.array(train_list)
y = data_csv.has_cactus
y = np.array(y)
densenet= keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet')
x = densenet.layers[-2].output

d = keras.layers.Dense(512,activation='relu')(x)

e = keras.layers.Dense(1,activation='sigmoid')(d)
model = keras.models.Model(densenet.input,e)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(train_list,y,validation_split=0.20,epochs=5)
testdf = pd.read_csv("../input/sample_submission.csv")

testx = []

for i in testdf.id:

    temp = np.array(Image.open('../input/test/test/'+i))

    reflect = cv2.copyMakeBorder(temp,96,96,96,96,cv2.BORDER_REFLECT)

    testx.append(reflect)

testx = np.array(testx)
result = model.predict(testx)
testdf.has_cactus = result
testdf.to_csv('submission.csv', index=False)