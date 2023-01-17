# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/diabetes.csv')

data.head()

data.isnull().sum()
y=data.Outcome

x_data=data.drop(['Outcome'],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T

x_train.shape
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train.T,y_train.T)

print('test accuracy {}'.format(lr.score(x_train.T,y_train.T)))