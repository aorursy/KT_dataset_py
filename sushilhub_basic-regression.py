# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
#import matplotlib.pylab as plt
import matplotlib.pyplot as plote

#from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import explained_variance_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
movie = pd.read_csv("../input/movie_dataset.csv")
movie.head()
movie.info()
movie.shape
movie.director_name.value_counts()
movie.country.value_counts()
movie.genres.value_counts()
movie.info()
list = ['budget', 'title_year', 'movie_facebook_likes', 'num_user_for_reviews', 'cast_total_facebook_likes', 'num_voted_users', 'gross', 'actor_1_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes', 'num_critic_for_reviews', 'duration', 'actor_2_facebook_likes', 'aspect_ratio']
movie.shape
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1,)
imputer = imputer.fit(movie[list])
movie[list] = imputer.transform(movie[list])

movie.info()
movie[list].head()
movie[list].shape
movie = movie.fillna(0)
movie.info()
movie.describe()
mlist = movie[list]
mlist.shape
mlist.head()
scaler = StandardScaler()
mlist = scaler.fit_transform(mlist)
movie[list] = mlist
movie[list].head()
movie.head()
X = movie[list]
X
Y = movie['imdb_score']
Y
Y = Y.reshape(-1, 1)
Y = scaler.fit_transform(Y)

Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
X_train.shape
X_test.shape
lr=LinearRegression()
lr.fit(X_train, y_train)
y_prediction = lr.predict(X_test)
y_prediction
y_test
print(metrics.mean_squared_error(y_test, y_prediction))
rms = sqrt(0.7908988818600056)
print("RMS value is", rms)
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)


xgb.fit(X_train,y_train)
xgb_prediction = xgb.predict(X_test)
rmsxgb = sqrt(metrics.mean_squared_error(y_test, xgb_prediction))
print("RMS of xgb is", rmsxgb)
lr_score_train = lr.score(X_test, y_test)
lr_score_test = lr.score(X_train, y_train)

print("Training score: ",lr_score_train)
print("Testing score: ",lr_score_test)

f,ax = plote.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


movie[list].info()
