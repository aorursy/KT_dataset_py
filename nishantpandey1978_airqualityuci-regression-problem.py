#importing required libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#read the dataset

dataset_name='AirQualityUCI'

dataset=pd.read_excel('../input/aqiuci/AirQualityUCI.xlsx',na_values='-200')
#check the dataset for null values

dataset.isnull().sum()
#segregating the dataset into Feature Matrix 'X' and Vector of Predictions 'Y'

X=dataset.drop(['Date','Time','AH'],axis=1)

y=np.array(dataset['AH'])
#import Simple Imputer from sklearn

from sklearn.impute import SimpleImputer

si=SimpleImputer(strategy='median')
#fill the blank values in the dataset

X=si.fit_transform(X)

y=si.fit_transform(y.reshape(-1,1))

X=pd.DataFrame(X)

X.isnull().sum()
#scaling the data

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()



X=ss.fit_transform(X)
#applying Linear Regression

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



lr.fit(X,y)

lr.score(X,y)