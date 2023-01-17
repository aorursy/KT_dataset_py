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
# importing libraries

import numpy as np

import pandas as pd

from pandas import datetime

from datetime import datetime

from datetime import date

import calendar

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
# shape of training and testing data

train.shape, test.shape
train.head()
test.head()
sn.heatmap(train.corr())
sn.heatmap(test.corr())
# distribution of count variable

sn.distplot(train["count"])
# distribution of count variable

sn.distplot(np.log(train["count"]))
sn.distplot(train["registered"])
# looking at the correlation between numerical variables

corr = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()

mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
# looking for missing values in the datasaet

train.isnull().sum()
# looking for missing values in the datasaet

test.isnull().sum()
# extracting date, hour and month from the datetime

train["date"] = train.datetime.apply(lambda x : x.split()[0])

train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0])

train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%d-%m-%Y").month)
train=pd.read_csv('../input/train.csv')
training = train[train['datetime']<='2012-03-30 0:00:00']

validation = train[train['datetime']>'2012-03-30 0:00:00']
test=pd.read_csv('../input/test.csv')
train = train.drop(['datetime', 'atemp'],axis=1)

test = test.drop(['datetime', 'atemp'], axis=1)

training = training.drop(['datetime', 'atemp'],axis=1)

validation = validation.drop(['datetime', 'atemp'],axis=1)
from sklearn.linear_model import LinearRegression
# initialize the linear regression model

lModel = LinearRegression()
X_train = training.drop('count', 1)

y_train = np.log(training['count'])

X_val = validation.drop('count', 1)

y_val = np.log(validation['count'])
# checking the shape of X_train, y_train, X_val and y_val

X_train.shape, y_train.shape, X_val.shape, y_val.shape
# fitting the model on X_train and y_train

lModel.fit(X_train,y_train)
# making prediction on validation set

prediction = lModel.predict(X_val)
prediction
from sklearn.tree import DecisionTreeRegressor
# defining a decision tree model with a depth of 5. You can further tune the hyperparameters to improve the score

dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train, y_train)
predict = dt_reg.predict(X_val)
# defining a function which will return the rmsle score

def rmsle(y, y_):

    y = np.exp(y),   # taking the exponential as we took the log of target variable

    y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
# calculating rmsle of the predicted values

rmsle(y_val, predict)
test_prediction = dt_reg.predict(test)
final_prediction = np.exp(test_prediction)
submission = pd.DataFrame()
# creating a count column and saving the predictions in it

submission['count'] = final_prediction