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
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print("Shape of Training data = ", train.shape)
print("Sahpe of Test Data = ", test.shape)

train.head()
test.head()
input.info()
test.info()
label = train.iloc[:, 0:1]
train = train.iloc[:, 1:]
print("trainX = ", train.shape, "trainY = ", label.shape)
#Variance is the average of all squared deviations.
#https://acadgild.com/blog/descriptive-statistics-standard-deviation-variance
#We are interested in removing those features whose varience is less or tends to 0
#As those features are far away from mean (of trand of main dataset) and will take unnecessary computation time
variance = np.var(train, axis = 0) >1000
train = train.loc[:, variance]
test = test.loc[:, variance]
print("Shape of Training data = ", train.shape)
print("Sahpe of Test Data = ", test.shape)

np.var(train, axis = 0)



variance






















