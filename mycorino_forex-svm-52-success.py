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
#!pip install iexfinance
# Machine learning

from sklearn.svm import SVC

from sklearn.metrics import scorer

from sklearn.metrics import accuracy_score



# For data manipulation

import pandas as pd

import numpy as np



# To plot

import matplotlib.pyplot as plt

import seaborn
# # Fetch the Data

# from iexfinance import get_historical_data 

# from datetime import datetime



# start = datetime(2017, 1, 1) # starting date: year-month-date

# end = datetime(2018, 1, 1) # ending date: year-month-date



# Df = get_historical_data('SPY', start=start, end=end, output_format='pandas')          

# Df= Df.dropna()

# Df = Df.rename (columns={'open':'Open', 'high':'High','low':'Low', 'close':'Close'})



# Df.Close.plot(figsize=(10,5))

# plt.ylabel("S&P500 Price")

# plt.show()

Df = pd.read_csv('../input/eur_dol_indicators5mnV2.csv')[['mid.o','mid.h','mid.l','mid.c']]
Df.head()
Df = Df.tail(10000)
Df= Df.dropna()

Df = Df.rename (columns={'mid.o':'Open', 'mid.h':'High','mid.l':'Low', 'mid.c':'Close'})



Df.Close.plot(figsize=(10,5))

plt.ylabel("eur/dol price")

plt.show()

y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)
Df['Open-Close'] = Df.Open - Df.Close

Df['High-Low'] = Df.High - Df.Low

X=Df[['Open-Close','High-Low']]

X.head()
split_percentage = 0.8

split = int(split_percentage*len(Df))



# Train data set

X_train = X[:split]

y_train = y[:split] 



# Test data set

X_test = X[split:]

y_test = y[split:]
cls = SVC().fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, cls.predict(X_train))

accuracy_test = accuracy_score(y_test, cls.predict(X_test))



print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))

print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))
Df['Predicted_Signal'] = cls.predict(X)

# Calculate log returns

Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)*100

Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal

Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))

plt.ylabel("Strategy Returns (%)")

plt.show()