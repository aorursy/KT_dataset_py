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
import pandas as pd

import numpy as np

import math

import datetime

from statistics import mean

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import datasets

import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self,learning_rate=0.001,n_iters=1000):

        self.lr=learning_rate

        self.n_iters=n_iters

        self.weights=None

        self.bias=None

    def fit(self,X,y):

        n_samples,n_features=X.shape

        self.weights=np.zeros(n_features)

        self.bias=0

        for _ in range(self.n_iters):

            y_predicted=np.dot(X,self.weights)+self.bias

            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))

            db=(1/n_samples)*np.sum(y_predicted-y)

            self.weights-=self.lr*dw

            self.bias-=self.lr*db

    def predict(self,X):

        y_approximated=np.dot(X,self.weights)+self.bias

        return y_approximated

        

df=pd.read_csv("../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv",index_col='Date',parse_dates=True)

df=df[['Open','High','Low','Close','Volume']]

df['HL_PCT']=(df['High']-df['Low'])/df['Low']*100.0

df['PCT_Change']=(df['Close']-df['Open'])/df['Open']*100.0

df=df[['Close','HL_PCT','PCT_Change','Volume']]

forecast_col='Close'

#df.drop(df.index[[1340,335]])

df.fillna(-99999,inplace=True)

print(len(df))

forecast_out=int(math.ceil(0.01*len(df)))

print(forecast_out)

df['label']=df[forecast_col].shift(-forecast_out)

#df.dropna(inplace=True)



X=np.array(df.drop(['label'],1))



X=preprocessing.scale(X)

X_lately=X[-forecast_out:]

X=X[:-forecast_out]



df.dropna(inplace=True)

y=np.array(df['label'])

print(len(X),len(y))

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(x_train.shape)

print(y_train.shape)

regressor=LinearRegression(learning_rate=0.001,n_iters=100000)

regressor.fit(x_train,y_train)

predictions=regressor.predict(x_test)

y1=sum((y_test-predictions)**2)

y_m=mean(y_test)

y2=sum((y_test-y_m)**2)

r2=1-(y1/y2)

print(r2)

forecast_set=regressor.predict(X_lately)

print(forecast_set)

df['Forecast']=np.nan

last_date=df.iloc[-1].name

last_unix=last_date.timestamp()

one_day=86400

next_unix=last_unix+one_day

for i in forecast_set:

    next_date=datetime.datetime.fromtimestamp(next_unix)

    next_unix+=one_day

    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

df['label'].plot()

df['Forecast'].plot()

plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()



#cmap=plt.get_cmap('viridis')

#fig=plt.figure(figsize=(8,6))

#m1=plt.scatter(x_train,y_train,color=cmap(0,9),s=10)

#m2=plt.scatter(x_test,y_test,color=cmap(0,5),s=10)

#plt.plot(X,y_pred_line,color='black',linewidth=2,label="Predictions")

#plt.show()