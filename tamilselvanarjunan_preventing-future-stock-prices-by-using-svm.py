# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import math,datetime
import time
import arrow
from sklearn import preprocessing,cross_validation,svm # Preprocessing for scaling data,Accuracy,Processing speed ,cross validation for training and testing
from sklearn.linear_model import LinearRegression #
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle

# Any results you write to the current directory are saved as output.
#Simple Linear regression works whatever the features we feed through it
#Read the data
df = pd.read_csv('../input/SHARADAR-Stock.csv')
print (df.head())
#Simplify the data to get High Accuracy
#We just re created the data frame by using below parameters
#here high and low describes the price change in day
#Open means starting price of the day
#close means tells the end prices of day
df=df[['open','high','low','close','volume']]
print (df.head())

#here high (highest stock price of the day) and close(the stock price at the end of the day)
#Calculating percent volatility
df['HIGHLOW_PCT']=(df['high']-df['close'])/(df['close'])*100
#Calculating new and old prices
df['PCT_Change']=(df['close']-df['open'])/(df['open'])*100
# Extracting required data from file
df=df[['close','HIGHLOW_PCT','PCT_Change','volume']]
print (df.head())
#forecast volume to calculate future stocks
forecast_col='close'
#We have to replace to na data with negative 99999.It will be useful when we lacking with data
df.fillna(-99999,inplace=True)
# if the length of data frame is returning decimal point or float it will round up to integer
# 0.1 means tomorrow data ,we can change accordingly
forecast_out=int(math.ceil(0.01*len(df)))
print (forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
print (df.head())

#print (df.tail())
#Setting up features and labels,X->feature and y->label
#drop useless features
#This code will return new data frame and convereted to 
#X_lately is the one we are predict against

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately=X[-forecast_out:]


df.dropna(inplace=True)
y=np.array(df['label'])
#print (len(X),len(y))
#Train and test data set
#Test size=0.2 means we are using 20% data as a testing data
#cross validation will take features and lables data and shuffle them and give X_train,y_train,X_test and y_test
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#classification inorder to get X_train and Y_train
#if i use Support vector machine i'm getting 80% Accuracy and  Linear regression i'm getting 75% Accuracy
#clf=LinearRegression(n_jobs=-1) =>75% Accuracy
clf=svm.SVR()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print (accuracy)
#We can pass single value or array of values or we are passing 99days of value
forecast_set=clf.predict(X_lately)
print (forecast_set,accuracy,forecast_out)
df['Forecast']=np.nan
last_date=df.iloc[-1].name
last_unix = arrow.get(last_date).timestamp
one_day=86400
next_unix=last_unix + one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i]

df['close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
    
#Forecast line is the predicted future price for 100 days from above graph