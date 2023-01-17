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
train=pd.read_csv('/kaggle/input/used-cars-price-prediction/train-data.csv')
train
test=pd.read_csv('/kaggle/input/used-cars-price-prediction/test-data.csv')
test
train.isnull().sum()
train['Name'].value_counts()
train['Location'].unique()
#first of all we have to remove the unwanted characters from the columns 

import re

train['Mileage']=train['Mileage'].str.replace(r'\D+','')
train['Mileage']=train['Mileage'].astype('float')
test['Mileage']=test['Mileage'].str.replace(r'\D+','')
test['Mileage']=test['Mileage'].astype('float')
#similarly for columns like engine and power
train['Power']=train['Power'].str.replace(r'\D+','')
test['Power']=test['Power'].str.replace(r'\D+','')
train['Power'].unique()
for i in train['Power']:

    if i=='':

        train['Power']=train['Power'].str.replace(i,'0')

for i in test['Power']:

    if i=='':

        test['Power']=test['Power'].str.replace(i,'0')

#filling the missing values in power column

train['Power']=train['Power'].fillna(method='ffill')
test['Power']=test['Power'].fillna(method='ffill')
train['Power']=train['Power'].astype('int64')
test['Power']=test['Power'].astype('int64')
#removing the chars from engine column

train['Engine']=train['Engine'].str.replace('CC','')
train['Engine']=train['Engine'].fillna(method='ffill')
train['Engine']=train['Engine'].astype('int64')
test['Engine']=test['Engine'].str.replace('CC','')

test['Engine']=test['Engine'].fillna(method='ffill')

test['Engine']=test['Engine'].astype('int64')
train['New_Price']=train['New_Price'].str.replace(r'\D+','')
test['New_Price']=test['New_Price'].str.replace(r'\D+','')
#checking for the further null values in the dataset

train.isnull().sum()
train['Mileage']=train['Mileage'].fillna(train['Mileage'].mean())
train['New_Price']=train['New_Price'].fillna(method='ffill')
train['New_Price']=train['New_Price'].astype('float')
test['New_Price']=test['New_Price'].fillna(method='ffill')
test['New_Price']=test['New_Price'].astype('float')
train['Seats']=train['Seats'].fillna(method='ffill')
test['Seats']=test['Seats'].fillna(method='ffill')
train.isnull().sum()
train=train.dropna()

test=test.dropna()
train.info()
#lets drop the unnecessary features

train=train.drop(['Name','Year'],axis=1)
train=train.drop(['Unnamed: 0'],axis=1)
test=test.drop(['Name','Year','Unnamed: 0'],axis=1)
#now preprocess the categorical features

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(train.columns):

    if train[i].dtype=='object':

        

        

        train[i]=le.fit_transform(train[i])
for i in list(test.columns):

    if test[i].dtype=='object':

        

        

        test[i]=le.fit_transform(test[i])
y=train['Price']

x=train.drop(['Price'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)
score_1
from sklearn.ensemble import RandomForestRegressor

rfg=RandomForestRegressor()

rfg.fit(x_train,y_train)

pred_2=rfg.predict(x_test)

score_2=r2_score(y_test,pred_2)
score_2
import matplotlib.pyplot as plt

plt.scatter(pred_2,y_test)

plt.show()
predictions=rfg.predict(x_test)

import seaborn as sns

sns.scatterplot(x=predictions.flatten(),y=y_test)
# from the figure we can conclude the model is predicting with good accuracy and the line is fitted quite well