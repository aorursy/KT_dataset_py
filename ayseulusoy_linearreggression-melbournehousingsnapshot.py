#import the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
#import the dataset

dataset=pd.read_csv('../input/melb_data.csv')

dataset.head()
dataset.shape
dataset.index
dataset.columns
#we are checking the missing values

dataset.isnull().sum()
#replace the Nan values with mean

dataset['Car'].mean()
dataset['Car']=dataset['Car'].replace(np.NaN,dataset['Car'].mean())
dataset['BuildingArea']=dataset['BuildingArea'].replace(np.NaN,dataset['BuildingArea'].mean())
dataset['YearBuilt']=dataset['YearBuilt'].replace(np.NaN,dataset['YearBuilt'].mean())
dataset.isnull().sum()
ca=dataset.iloc[ : , :-1].values

ca
#'CouncilArea' is a big problem so we'll convert into numerical values

from sklearn.preprocessing import LabelEncoder 

le=LabelEncoder()
ca[ : ,0]=le.fit_transform(ca[ :,0])

ca
dataset['CouncilArea']=dataset['CouncilArea'].replace(np.NaN,0)
dataset.isnull().sum() #no more missing value
#splitting the dataset into Training and Test set 

rn=dataset['Regionname'] #we are converting the region name to numeric values bec we'll draw figure of regionname and price

rn=le.fit_transform(rn)
#so firstly,we are drawing figure of dataset to understand easily dataset

plt.scatter(rn,dataset['Price'])
X=dataset['Regionname']



Y=dataset['Price']
X
rn
rn.reshape(-1, 1)
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(rn,Y,test_size=0.2) #%80 train,%20 test.It is choosing randomly
len(X_train)
len(X_test)
X_train=X_train.reshape(-1,1)

X_train
X_test=X_test.reshape(-1,1)#we can checking to random selection

X_test
len(Y_train)
len(Y_test)
Y_test
#now,we'll do linear regression

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test)
Y_test
lr.score(X_test,Y_test) #Accuracy is 0.0033%