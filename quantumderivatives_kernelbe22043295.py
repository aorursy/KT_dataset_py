# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Machine Learning

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

#for data manipulation

import pandas as pd

import numpy as np

#To plot

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#To ignore warnings

import warnings

warnings.filterwarnings("ignore")
#Readthe XRP Historical Data

#We read the XRP Historical Data from CSV file

#Fetch the data

Df=pd.read_csv("../input/XRP Historical Data - Investing.com.csv")

Df=Df.dropna()

Df=Df.set_index(Df.Date)

Df=Df.drop(columns='Date')

Df.head()
#Define the explanatory variables

#Predictor Variables

Df['Open-Low']=Df.Open-Df.Low

Df['High-Low']=Df.High-Df.Low

X=Df[['Open-Low','High-Low']]

X.head()
#Define the target variables

#Target variables

y=np.where(Df['Low'].shift(-1)>Df['Low'],1,-1)

#Split the data into train and test

split_percentage=0.8

split=int(split_percentage*len(Df))

#Train the dataset

X_train=X[:split]

y_train=y[:split]

#Test data set

X_test=X[split:]

y_test=y[split:]
#Support Vector Classifier(SVC)

#Support Vector Classifier

cls=SVC().fit(X_train,y_train)

#Classifier Accuracy

#Train and test Accuracy

accuracy_train=accuracy_score(y_train,cls.predict(X_train))

accuracy_test=accuracy_score(y_test,cls.predict(X_test))

print('\nTrain Accuracy:{:.2f}%'.format(accuracy_train*100))

print('Test Accuracy:{:.2f}%'.format(accuracy_test*100))
#Startegy Implementation

#Predicted Signal

Df['Predicted_Signal']=cls.predict(X)

#Calculate daily returns

Df['Return']=Df.Low.pct_change()

#Calculate strategy returns

Df['Strategy_Return']=Df.Return*Df.Predicted_Signal

#Calculate geometric returns

geometric_returns=(Df.Strategy_Return.iloc[split:]+1).cumprod()

#plot geometric returns

geometric_returns.plot(figsize=(10,5))

plt.ylabel("Strategy Returns(%)")

plt.Xlabel("Date")

plt.show()
