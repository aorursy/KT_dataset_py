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
#import required libraries

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#load NFLX data from dataset

df = pd.read_csv('../input/netflix-stock-price/NFLX.csv')

df.head(6)
# See number of data

df.shape
#Visualize the close price data

plt.figure(figsize=(16,8))

plt.title('Netflix')

plt.xlabel('Days')

plt.ylabel('Close Price USD ($)')

# plt.plot(df['Open'])

# plt.plot(df['High'])

# plt.plot(df['Low'])

plt.plot(df['Close'])

plt.show()
#convert date String into Integer

df['Date'].head()[0]



df['Date'] = df['Date'].str.replace('-','').astype(int)



df['Date'].head()[0]
#split data between train and test 

features = df.drop(['Close','Adj Close'],axis=1)

labels = df['Close']

# X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=30, test_size=0.2)



X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=30, test_size=0.2)

print(X_train.shape)

print(X_test.shape)
#now we are ready to declare regression model

lr = LinearRegression()



#training model

lr.fit(X_train,y_train)
#now we will predict result based on test data.



y_predict = lr.predict(X_test)



score = lr.score(X_test,y_predict)
#lets put orignal price an predicted price on graph

plt.figure(figsize=(16,8))

plt.title('Netflix prediction')

plt.xlabel('Days')

plt.ylabel('Close Price USD ($)')

plt.plot(y_predict)

plt.plot(y_predict,"o")

plt.show()
X_train

exp_data = [[20200723,490.24,490.91,476.75,437000000]]

exp_pred = lr.predict(exp_data)



print(exp_pred)

# 489.223224938