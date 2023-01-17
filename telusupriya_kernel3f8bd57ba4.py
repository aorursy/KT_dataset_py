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
from sklearn import datasets

from sklearn.svm import SVC

from scipy import misc

import numpy as np



digits = datasets.load_digits()

#print digits dataset

#print(digits)



features = digits.data

labels = digits.target

#printing data -features and target-labels

print(features)

print(labels)



clf = SVC(gamma = 0.001)

clf.fit(features,labels)

print(features.shape)



print(clf.predict([features[-2]]))

img = misc.imread('digits.jpg')



img = misc.imresize(img,(8,8))

print(img.dtype)

print(digits.images.dtype)

img = img.astype(digits.images.dtype)

print(img.dtype)

img = misc.bytescale(img,high = 16,low =0)

print(img.dtype)

print(features[-2])

print(img)



x_test = []

for eachRow in img:

    for eachPixel in eachRow:

        x_test.append(sum(eachPixel)/3.0)







print(clf.predict([x_test]))








