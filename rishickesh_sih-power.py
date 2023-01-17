# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import datetime as dt

from keras.models import Sequential

from keras.layers import Dense

import datetime as dt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ndata=pd.read_csv("../input/energy-usage-2010.csv")
fig, ax = plt.subplots(figsize=(60,60))



sns.heatmap(ndata.corr(),annot = True,cmap="YlGnBu")
ndata.columns
null_columns=ndata.columns[ndata.isnull().any()]

ndata[ndata["ELECTRICITY ACCOUNTS"].isnull()][null_columns].head()
ndata['ELECTRICITY ACCOUNTS'][ndata['ELECTRICITY ACCOUNTS']=="Less than 4"]=2

ndata.head()
ndata=ndata.drop(['KWH TOTAL SQFT','THERMS TOTAL SQFT', 'KWH STANDARD DEVIATION 2010', 'KWH MINIMUM 2010','KWH 1ST QUARTILE 2010', 'KWH 2ND QUARTILE 2010','KWH 3RD QUARTILE 2010', 'KWH MAXIMUM 2010', 'KWH SQFT MEAN 2010','KWH SQFT STANDARD DEVIATION 2010', 'KWH SQFT MINIMUM 2010','KWH SQFT 1ST QUARTILE 2010', 'KWH SQFT 2ND QUARTILE 2010','KWH SQFT 3RD QUARTILE 2010', 'KWH SQFT MAXIMUM 2010','THERM MEAN 2010', 'THERM STANDARD DEVIATION 2010','THERM MINIMUM 2010', 'THERM 1ST QUARTILE 2010','THERM 2ND QUARTILE 2010', 'THERM 3RD QUARTILE 2010','THERM MAXIMUM 2010','THERMS SQFT MEAN 2010','THERMS SQFT STANDARD DEVIATION 2010', 'THERMS SQFT MINIMUM 2010','THERMS SQFT 1ST QUARTILE 2010', 'THERMS SQFT 2ND QUARTILE 2010','THERMS SQFT 3RD QUARTILE 2010', 'THERMS SQFT MAXIMUM 2010','AVERAGE STORIES','AVERAGE BUILDING AGE','CENSUS BLOCK','OCCUPIED HOUSING UNITS'],axis=1)
ndata=ndata.dropna(subset=['BUILDING TYPE',

       'BUILDING_SUBTYPE'])

ndata.isna().sum()
ndata['KWH JANUARY 2010'].fillna(ndata['KWH JANUARY 2010'].median(),inplace=True)

ndata['KWH FEBRUARY 2010'].fillna(ndata['KWH FEBRUARY 2010'].median(),inplace=True)

ndata['KWH MARCH 2010'].fillna(ndata['KWH MARCH 2010'].median(),inplace=True)

ndata['KWH APRIL 2010'].fillna(ndata['KWH APRIL 2010'].median(),inplace=True)

ndata['KWH MAY 2010'].fillna(ndata['KWH MAY 2010'].median(),inplace=True)

ndata['KWH JUNE 2010'].fillna(ndata['KWH JUNE 2010'].median(),inplace=True)

ndata['KWH JULY 2010'].fillna(ndata['KWH JULY 2010'].median(),inplace=True)

ndata['KWH AUGUST 2010'].fillna(ndata['KWH AUGUST 2010'].median(),inplace=True)

ndata['KWH SEPTEMBER 2010'].fillna(ndata['KWH SEPTEMBER 2010'].median(),inplace=True)

ndata['KWH OCTOBER 2010'].fillna(ndata['KWH OCTOBER 2010'].median(),inplace=True)

ndata['KWH NOVEMBER 2010'].fillna(ndata['KWH NOVEMBER 2010'].median(),inplace=True)

ndata['KWH DECEMBER 2010'].fillna(ndata['KWH DECEMBER 2010'].median(),inplace=True)

ndata['ELECTRICITY ACCOUNTS'].fillna(ndata['ELECTRICITY ACCOUNTS'].median(),inplace=True)

#ndata['ELECTRICITY ACCOUNTS'].fillna(ndata['ELECTRICITY ACCOUNTS'].median(),inplace=True)
ndata.head()
ndata.isna().sum()
ndata['TOTAL KWH']=ndata['KWH JANUARY 2010']+ndata['KWH FEBRUARY 2010']+ndata['KWH MARCH 2010']+ndata['KWH APRIL 2010']+ndata['KWH MAY 2010']+ndata['KWH JUNE 2010']+ndata['KWH JULY 2010']+ndata['KWH AUGUST 2010']+ndata['KWH SEPTEMBER 2010']+ndata['KWH OCTOBER 2010']+ndata['KWH NOVEMBER 2010']+ndata['KWH DECEMBER 2010']
ndata.head()
ndata['THERM JANUARY 2010'].fillna(ndata['THERM JANUARY 2010'].median(),inplace=True)

ndata['THERM FEBRUARY 2010'].fillna(ndata['THERM FEBRUARY 2010'].median(),inplace=True)

ndata['THERM MARCH 2010'].fillna(ndata['THERM MARCH 2010'].median(),inplace=True)

ndata['TERM APRIL 2010'].fillna(ndata['TERM APRIL 2010'].median(),inplace=True)

ndata['THERM MAY 2010'].fillna(ndata['THERM MAY 2010'].median(),inplace=True)

ndata['THERM JUNE 2010'].fillna(ndata['THERM JUNE 2010'].median(),inplace=True)

ndata['THERM JULY 2010'].fillna(ndata['THERM JULY 2010'].median(),inplace=True)

ndata['THERM AUGUST 2010'].fillna(ndata['THERM AUGUST 2010'].median(),inplace=True)

ndata['THERM SEPTEMBER 2010'].fillna(ndata['THERM SEPTEMBER 2010'].median(),inplace=True)

ndata['THERM OCTOBER 2010'].fillna(ndata['THERM OCTOBER 2010'].median(),inplace=True)

ndata['THERM NOVEMBER 2010'].fillna(ndata['THERM NOVEMBER 2010'].median(),inplace=True)

ndata['THERM DECEMBER 2010'].fillna(ndata['THERM DECEMBER 2010'].median(),inplace=True)
ndata['TOTAL THERMS']=ndata['THERM JANUARY 2010']+ndata['THERM FEBRUARY 2010']+ndata['THERM MARCH 2010']+ndata['TERM APRIL 2010']+ndata['THERM MAY 2010']+ndata['THERM JUNE 2010']+ndata['THERM JULY 2010']+ndata['THERM AUGUST 2010']+ndata['THERM SEPTEMBER 2010']+ndata['THERM OCTOBER 2010']+ndata['THERM NOVEMBER 2010']+ndata['THERM DECEMBER 2010']
ndata.isna().sum()
ndata=ndata.dropna(subset=['TOTAL POPULATION'])
ndata.isna().sum()
ndata=ndata.drop(['KWH MEAN 2010','OCCUPIED UNITS PERCENTAGE','RENTER-OCCUPIED HOUSING PERCENTAGE'],axis=1)
ndata.head()
ndata.shape
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()



from sklearn.preprocessing import scale

colu=['COMMUNITY AREA NAME','BUILDING TYPE','BUILDING_SUBTYPE']



for col in colu:

    ndata[col] = le.fit_transform(ndata[col])
ndata.head()
ndata=pd.get_dummies(ndata,columns=['BUILDING TYPE','BUILDING_SUBTYPE'])
ndata.head()
X=ndata.iloc[:, [0,14,15,30,32,33,35,36,37,38,39,40,41,42,43]].values

y=ndata.iloc[:, 1].values
from sklearn.model_selection import train_test_split

#split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test) 
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()

mlr.fit(X_train, y_train)



#predict the test set results

y_pred = mlr.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(y_test, y_pred))



from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



rmse, r2, adj_r2
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='distance')

knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)



from sklearn.metrics import mean_squared_error

from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



r2, adj_r2