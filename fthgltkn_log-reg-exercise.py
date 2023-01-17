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
data=pd.read_csv('../input/advertising.csv')

data.head()
data.columns=['Daily_Time_Spent_on_Site', 'Age', 'Area Income',

       'Daily_Internet_Usage', 'Ad_Topic_Line', 'City', 'Male', 'Country',

       'Timestamp', 'Clicked_on_Ad']
data.drop(['Ad_Topic_Line','City','Timestamp'],axis=1,inplace=True)
data.drop(['Country'],axis=1,inplace=True)
data.tail()
data.info()
data.isnull().sum()
x=data.Clicked_on_Ad

y=data.drop(['Clicked_on_Ad'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(y,x,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

print('test accuracy {}'.format(lr.score(x_train,y_train)))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,lr.predict(y_test))

print(cm)