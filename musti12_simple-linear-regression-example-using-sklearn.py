# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bcd=pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")

bcd.info()

y=bcd.iloc[:,5].values #for my binary values

x_data=bcd.drop(["diagnosis"],axis=1) #to use this in normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) #normalization
print(x_data)
print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T

y_train=y_train.T

x_test=x_test.T

y_test=y_test.T
print(x_train)
print(y_train)
print(x_test)
print(y_test)
lr=LinearRegression()

lr.fit(x_train.T,y_train.T)
print("Test accuracy: {}".format(lr.score(x_test.T,y_test.T)))
