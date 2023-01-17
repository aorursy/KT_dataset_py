import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os



train = []

for dirname, _, filenames in os.walk('../input/chest-xray/chest_xray/train/Covid19'):

    for filename in filenames:

        myFile = os.path.join(dirname, filename)

        image = cv2.resize(cv2.imread (myFile,0),(128,128))

        image = image[:,:,np.newaxis]

        train.append(image)





train_covid_128 = np.array(train,dtype='float32')

train_covid_128.shape

test = []

for dirname, _, filenames in os.walk('../input/chest-xray/chest_xray/test/Covid19'):

    for filename in filenames:

        myFile = os.path.join(dirname, filename)

        img = cv2.resize(cv2.imread (myFile,0),(128,128))

        img = img[:,:,np.newaxis]

        test.append(img)





test_covid_128 = np.array(test,dtype='float32')

test_covid_128.shape

np.save('train_covid_128',train_covid_128)

data=np.load('train_covid_128.npy')

data.shape

np.save('test_covid_128',test_covid_128)

data=np.load('test_covid_128.npy')

data.shape
