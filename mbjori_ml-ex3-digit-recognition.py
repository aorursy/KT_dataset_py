# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy.io

mat = scipy.io.loadmat('/kaggle/input/ex3data1.mat')
# i = 3456
# imgPix = mat['X'][i]
# print (mat['y'][i])
# imgPix = np.array(imgPix).reshape((20,20)).T

# from matplotlib import pyplot as plt
# plt.imshow(imgPix, interpolation='nearest')

# plt.show()

from sklearn.model_selection import train_test_split
X = mat['X']
y = mat['y']

theta0 = np.ones((X.shape[0],1))

X = np.concatenate((theta0,X), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
def sigmoid(val):
    return 1/ (1 + np.exp(-val))
K = np.unique(y)
iterations = 100000
m = X_train.shape[0]
alpha = 0.01

tempY = y_train

theta = np.zeros((X_train.shape[1],K.shape[0]))
# print (theta[:,1].shape)
for i in range(K.shape[0]):
    tempY[tempY == K[i]] = 1
    tempY[tempY != K[i]] = 0
    
    tempTheta = theta[:,i].reshape((X_train.shape[1],1))
#     print (tempTheta.shape)
        
    for _ in range(iterations):
        hyp = np.apply_along_axis(sigmoid,arr = np.dot(X_train, tempTheta), axis = 1)
        tempTheta = tempTheta - (1/m * np.sum(np.multiply(hyp - tempY, X_train), axis = 0).reshape(X_train.shape[1],1))
    theta[:,i] = tempTheta[:,0]

    tempYTest = y_test
    tempYTest[tempYTest != K[i]] = 0
    tempYTest[tempYTest == K[i]] = 1
    
    tempTheta = theta[:,i].reshape((X_train.shape[1],1))

    hypTest = np.apply_along_axis(sigmoid, arr = np.dot(X_test, tempTheta), axis = 1)
    error = np.sum(np.absolute(np.round(hypTest) - tempYTest))
    
    print (((1500 - error)/ 1500) * 100)
    

tempTheta = theta[:,i].reshape((X_train.shape[1],1))

hypTest = np.apply_along_axis(sigmoid, arr = np.dot(X_test, tempTheta), axis = 1)

# hypTest[ hypTest >= 0.5] = 1
# hypTest[ hypTest < 0.5] = 0

print (np.sum(np.absolute(np.round(hypTest) - tempYTest)))
((1500-19)/1500) * 100