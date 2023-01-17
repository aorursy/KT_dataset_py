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
# importing Libraries


import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plote
import seaborn as sns
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pylab import subplot, savefig


train=pd.read_csv("../input/GDSChackathon.csv")
test=pd.read_csv("../input/GDSChackathon.csv")
train.head()
train.describe()
train = train.drop(['genres', 'plot_keywords'], axis=1)
train.info()
train['director_facebook_likes']=train['director_facebook_likes'].fillna((train['director_facebook_likes'].mean()))
train['num_critic_for_reviews']=train['num_critic_for_reviews'].fillna((train['num_critic_for_reviews'].mean()))
train['duration']=train['duration'].fillna((train['duration'].mean()))
train['actor_3_facebook_likes']=train['actor_3_facebook_likes'].fillna((train['actor_3_facebook_likes'].mean()))
train['actor_1_facebook_likes']=train['actor_1_facebook_likes'].fillna((train['actor_1_facebook_likes'].mean()))
train['gross']=train['gross'].fillna((train['gross'].mean()))
train['num_voted_users']=train['num_voted_users'].fillna((train['num_voted_users'].mean()))
train['cast_total_facebook_likes']=train['cast_total_facebook_likes'].fillna((train['cast_total_facebook_likes'].mean()))
train['facenumber_in_poster']=train['facenumber_in_poster'].fillna((train['facenumber_in_poster'].mean()))
train['num_user_for_reviews']=train['num_user_for_reviews'].fillna((train['num_user_for_reviews'].mean()))
train['budget']=train['budget'].fillna((train['budget'].mean()))
train['title_year']=train['title_year'].fillna((train['title_year'].mean()))
train['actor_2_facebook_likes']=train['actor_2_facebook_likes'].fillna((train['actor_2_facebook_likes'].mean()))
train['aspect_ratio']=train['aspect_ratio'].fillna((train['aspect_ratio'].mean()))
train['movie_facebook_likes']=train['movie_facebook_likes'].fillna((train['movie_facebook_likes'].mean()))




train.info()
scale_list = ['director_facebook_likes','actor_1_facebook_likes','aspect_ratio','actor_2_facebook_likes','actor_3_facebook_likes','gross','num_voted_users','budget','cast_total_facebook_likes','movie_facebook_likes']
sc = train[scale_list]
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
train[scale_list] = sc
train[scale_list].head()
train['color'].fillna('color', inplace=True)
train['director_name'].fillna('abc',inplace=True)
train['actor_2_name'].fillna('abc',inplace=True)
train['actor_3_name'].fillna('abc',inplace=True)
train['actor_1_name'].fillna('abc',inplace=True)
train['movie_title'].fillna('abc',inplace=True)
train['movie_imdb_link'].fillna('abc',inplace=True)
train['language'].fillna('abc',inplace=True)
train['country'].fillna('abc',inplace=True)
train['content_rating'].fillna('abc',inplace=True)
train.info()
encoding_list = ['color', 'director_name','actor_2_name','actor_3_name','actor_1_name', 'movie_title','movie_imdb_link','language','country','content_rating']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)
train.head()
y = train['imdb_score']
x = train.drop('imdb_score', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
X_test.shape
logreg=LinearRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
y_test
y_pred
print(sqrt(metrics.mean_squared_error(y_test, y_pred)))
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)
print(sqrt(metrics.mean_squared_error(y_test, predictions)))
features = ['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross','director_facebook_likes',
       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'budget', 'title_year',
       'actor_2_facebook_likes', 'aspect_ratio',
       'movie_facebook_likes','num_critic_for_review','duration',]
target = ['imdb_score']
f,ax = plote.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.reset_orig()
sns.pairplot(train[['cast_total_facebook_likes','movie_facebook_likes','content_rating']],hue='content_rating',palette='inferno')

