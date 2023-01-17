# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler

from sklearn.linear_model import LogisticRegression,SGDClassifier,LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

import keras

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Dense,LSTM

import tensorflow as tf
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test= pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
train['Province_State'].fillna("",inplace = True)

test['Province_State'].fillna("",inplace = True)
train['Country_Region'] = train['Country_Region'] + ' ' + train['Province_State']

test['Country_Region'] = test['Country_Region'] + ' ' + test['Province_State']
train.drop("Province_State",axis=1,inplace=True)

test.drop("Province_State",axis=1,inplace=True)
def extractmonth(x):

    ans=x.split("-")

    return ans[1]







def extractday(x):

    ans=x.split("-")

    return ans[2]
train["Month"]=train["Date"].map(extractmonth)

test["Month"]=test["Date"].map(extractmonth)



train["Day"]=train["Date"].map(extractday)

test["Day"]=test["Date"].map(extractday)
train["Month"]=train["Month"].astype(int)

test["Month"]=test["Month"].astype(int)





train["Day"]=train["Day"].astype(int)

test["Day"]=test["Day"].astype(int)
lb = LabelEncoder()

train['Country_Region'] = lb.fit_transform(train['Country_Region'])

test['Country_Region'] = lb.transform(test['Country_Region'])
plt.figure(figsize = (10,10))

corr = train.corr()

sns.heatmap(corr , mask=np.zeros_like(corr, dtype=np.bool) , cmap=sns.diverging_palette(-100,0,as_cmap=True) , square = True)
!pip install pycountry_convert
train.drop("Date",axis=1,inplace=True)

test.drop("Date",axis=1,inplace=True)
Id=test["ForecastId"]
test.drop("ForecastId",axis=1,inplace=True)

train.drop("Id",axis=1,inplace=True)
X=train.drop(["Fatalities","ConfirmedCases"],axis=1)

y=train["ConfirmedCases"]

z=train["Fatalities"]
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=10000,

    learning_rate=0.1,

    depth=9,random_strength=10,l2_leaf_reg=4,bagging_temperature=0.5)
model.fit(X,y)
Cc=model.predict(test)
model.fit(X,z)
fet=model.predict(test)
rf = XGBRegressor(n_estimators = 24000 , random_state =101, max_depth = 24)

rf.fit(X,y)
CC=rf.predict(test)
rf.fit(X,z)
fat=rf.predict(test)
sub=pd.DataFrame()

sub["ForecastId"]=Id

sub["ConfirmedCases"]=Cc

sub["Fatalities"]=fet

sub.to_csv("submission.csv",index=False)