# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load data set

x_l = np.load('../input/sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(x_l[1200].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis('off')
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1) 

print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

print(number_of_train)

print(number_of_test)
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)