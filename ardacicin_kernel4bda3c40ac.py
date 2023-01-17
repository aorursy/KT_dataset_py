import numpy as np 

from scipy.stats import norm #Analysis 

import pandas as pd 

import os

from random import sample 

from scipy import stats #Analysis

import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt

import matplotlib



print(os.listdir("../input"))
train_csv = pd.read_csv('../input/train.csv')

test_csv = pd.read_csv('../input/test.csv')

print (train_csv.shape)

print (train_csv.shape)

train_csv.head()



test_csv.head()




print("skewness: %f" % train_csv['SalePrice'].skew())

print("kurtosis: %f" % train_csv['SalePrice'].kurt())



fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(train_csv['SalePrice'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(train_csv['SalePrice']), plot=plt)