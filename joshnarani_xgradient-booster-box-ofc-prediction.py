import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from collections import OrderedDict



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
#loading the data

train=pd.read_csv('../input/train.csv')
#loading the data

test=pd.read_csv('../input/test.csv')
#dropping few columns

train.drop(["homepage","imdb_id","belongs_to_collection","tagline","status","original_title","overview","genres","poster_path","production_countries", "production_companies","release_date","spoken_languages","title",'Keywords', 'cast', 'crew'], axis = 1,inplace=True)

test.drop(["homepage","imdb_id","belongs_to_collection","tagline","status","original_title","overview","genres","poster_path","production_countries", "production_companies","release_date","spoken_languages","title",'Keywords','cast', 'crew'], axis =1,inplace = True)
from sklearn.preprocessing import LabelEncoder

#label encode original language



le = LabelEncoder()

train['original_language'] = le.fit_transform(train['original_language'])

train.head(1)
test['original_language'] = le.fit_transform(test['original_language'])

test.head(1)
train.head()
#Imputing missing values for both train and test

train.fillna(train['runtime'].mode() ,inplace=True)

test.fillna(test['runtime'].mode()  ,inplace=True)

df=pd.DataFrame()

df=pd.concat([test,train],sort=False) 
df.tail()
df.fillna(df['revenue'].mean(),inplace=True)
df.head()
df.tail()
X = df.drop(['revenue'],axis=1)

y = df.revenue



from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.7, random_state=0)
from xgboost import XGBRegressor
# fit model no training data

model = XGBRegressor()

model.fit(X_train, y_train)
pr=model.predict(test)
pr.shape
submission = pd.DataFrame({'id': test.id , 'revenue': pr})

submission.tail()
submission.head()
submission[3401:3600]
submission.to_csv("submission.csv",index=False)