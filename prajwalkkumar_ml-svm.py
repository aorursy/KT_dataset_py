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
import cv2

import numpy as np

import pandas as pd

import mahotas

import h5py

from sklearn.preprocessing import MinMaxScaler

traindf=pd.read_csv("../input/train_labels.csv",dtype=str)

testdf=pd.read_csv("../input/sample_submission.csv",dtype=str)

combineddf = pd.concat([traindf,testdf])
def fd_hu_moments(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = cv2.HuMoments(cv2.moments(image)).flatten()

    return feature

def fd_haralick(image):    # convert the image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the haralick texture feature vector

    haralick = mahotas.features.haralick(gray).mean(axis=0)

    return haralick

 

def fd_histogram(image, mask=None):

    # convert the image to HSV color-space

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bins = 256

    # compute the color histogram

    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    # normalize the histogram

    cv2.normalize(hist, hist)

    hist= hist.flatten()

    return hist
global_features = []

for i in range(len(traindf)):

    image = cv2.imread("../input/train/"+traindf.id[i]+".tif")

    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    global_features.append(global_feature)

for i in range(len(testdf)):

    image = cv2.imread("../input/test/"+testdf.id[i]+".tif")

    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    global_features.append(global_feature)

#Normalize The feature vectors...

rescaled_features = scaler.fit_transform(global_features)

target = list(traindf.label)
h5f_data = h5py.File('data.h5', 'w')

h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))



h5f_label = h5py.File('labels.h5', 'w')

h5f_label.create_dataset('dataset_1', data=np.array(target))



h5f_data.close()

h5f_label.close()