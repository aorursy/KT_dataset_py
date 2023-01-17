# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import svm

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/fashion-mnist_train.csv', header=0)

test = pd.read_csv('../input/fashion-mnist_test.csv', header=0)
train.head()


train_2 = train.drop("label", axis=1)

train_2=train_2.iloc[:,:]

train_lab=train.iloc[:,:1]



#train_2.shape

train_lab.shape

type(train_lab.values)
test_2= test.drop("label", axis=1)

test_lab=test.iloc[:,:1]
train_im=train_2.iloc[3,:]

train_im=train_im.values

train_im=train_im.reshape((28,28))

#img=train_2.iloc[3].as_matrix()

#img=img.reshape((28,28))

#plt.imshow(train_2.iloc[3].values[1:].reshape((28, 28)))

plt.imshow(train_im)
#train_im.shape

#type(train_im)

#train_fn=train_im.reshape(1,784)

#train_fn.shape
clf = svm.SVC(gamma=0.1, kernel='poly')

clf.fit(train_2.values, train_lab.values)

train_2.values.shape
train_lab.values.shape
test_2.values.shape
test_lab.values.shape
clf.score(test_2.values,test_lab.values)
test_2.iloc[67,:].values.shape
clf.predict(test_2.iloc[72,:].values.reshape(1,-1))
test_lab.iloc[72]