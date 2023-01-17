# importing modules

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

% matplotlib inline

import seaborn as sns
# reading textfile using pandas

data = pd.read_csv('../input/corn_OHLC2013-2017.txt', sep=",", header=None)



# Giving columns Name with resspect to Dataset

data.columns =['Date', 'Open Price', 'High Price', 'Low Price', 'Close Price']

data.head()
# Describing Data

data.describe()
# Checking is there any null value 

data.isnull().sum()
# creating new columns data frame week nymber

data['week number'] = [i for i in range(1,len(data['Date'])+1)]

data.tail()
# Checking Correlation of features

data.corr()
# plotting Price realtion using Facegrid

sns.factorplot(data=data[['Open Price','High Price','Low Price','Close Price']])
# Creating Train And Target Data



# train Data

week = data['week number'].values.reshape(len(data['week number']),1)

# Target Data

Open_Price = data['Open Price']

High_Price = data['High Price']

Low_Price = data['Low Price']

Close_Price = data['Close Price'] 
# Creating SVR model for prediction of prices

from sklearn.svm import SVR



def SVResgressor(train,target):

    svr = SVR(C = 1e3,gamma = 0.1)

    svr.fit(train,target)

    pred = svr.predict(train)

    

    # plotting prediction and real values

    plt.figure(figsize=(15,5))

    plt.scatter(train,target,label ='Real Price')

    plt.plot(train,pred,'r',label ='Predicted Pricee')

    plt.legend()
# Model on Open Price

SVResgressor(week,Open_Price)
# Model on High Price

SVResgressor(week,High_Price)
# Model on Low price

SVResgressor(week,Low_Price)
# Model on Close Price

SVResgressor(week,Close_Price)