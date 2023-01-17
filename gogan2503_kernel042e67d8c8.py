# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
bikedf=pd.read_csv("../input/bike_share.csv")

bikedf1=bikedf.copy()

bikedf1.head()
#To find no.of rows and columns

bikedf1.shape
#Listing the columns of dataset

bikedf1.columns
#Dataframe.Info() This method prints information about a DataFrame including the

# index dtype and column dtypes, non-null values and memory usage

bikedf1.info()
#To detect any missing values

bikedf1.isna().sum()
#check the duplicated records

bikedf1.duplicated().sum()
#To return the duplicated rows

dfDuplicatedRows=bikedf1[bikedf1.duplicated()]

dfDuplicatedRows
#Check the count of duplicate records and remove duplicate records

bikedf1.duplicated().sum()

bikedf1.drop_duplicates(inplace=True)
#To check whether duplicated rows are removed

bikedf1.duplicated().sum()
#finding the numerical column

numcol=bikedf1.select_dtypes(include=np.number).columns

numcol
#Finding the Category column.

#Here no cat column all columns are numeric

catcol=bikedf1.select_dtypes(exclude=np.number).columns

catcol
bikedf1.corr()
#Heatmap to find the correlation between independent and dependent variable (count)

#Both temp and atemp features are higly influencing compare to other features

#Season is the next highly influencing  feature

plt.figure(figsize=(10,5))

ax=sns.heatmap(bikedf1.corr(),annot=True)

plt.show(ax)

#Find the outliers of temp feature.

#No outliers present

sns.boxplot(bikedf1['temp'])
#Find the outliers of air temp feature.

#No outliers present

sns.boxplot(bikedf1['atemp'])
#Find the outliers of season feature.

#No outliers present

sns.boxplot(bikedf1['season'])
#Loading y with dependant variable

y=bikedf1["count"]
#Filtering the dependant variable and loading X with Independant variable

X=bikedf1.drop(columns=["count",'holiday','workingday','weather','humidity','windspeed','casual',

               'registered'],axis=1)

X.head()
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=1)



print("Train X :", train_X.shape)

print("Test X :",test_X.shape)

print("Train y :",train_y.shape)

print("Test y :",test_y.shape)


model=LinearRegression()

model.fit(train_X,train_y)

print("Coefficient value:",model.coef_)

print("Intercept value:", model.intercept_)
train_y_predict=model.predict(train_X)

print(train_y_predict)

test_y_predict=model.predict(test_X)

print(test_y_predict)
#Predicting the Y value from the train set and test set

train_y_predict=model.predict(train_X)

display(train_y_predict)

test_y_predict=model.predict(test_X)

display(test_y_predict)

print("****************MAE***********************")

print("Mean Absolute Error Train: ",mean_absolute_error(train_y_predict,train_y))

print("Mean Absolute Error Test: ",mean_absolute_error(test_y_predict,test_y))



print('***********************MSE****************')

print("mean_squared_error Train: ", mean_squared_error(train_y_predict,train_y))

print("mean_squared_error Test: ", mean_squared_error(test_y_predict,test_y))

print

print('*********************RMSE******************')

print(np.sqrt(mean_squared_error(train_y_predict,train_y)))

print(np.sqrt(mean_squared_error(test_y_predict,test_y)))
