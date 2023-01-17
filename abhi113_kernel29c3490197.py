import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/GDSChackathon.csv")
display(train.head())

train.tail(10)
import scipy.stats as st
y = train['imdb_score']
plt.figure(1); plt.title('Normal')
sns.distplot(train['imdb_score']);
#plt.figure(2); plt.title('LogNormal')
#sns.distplot(train['imdb_score'], fit=st.lognorm);
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()  
train.shape
train.info()
train.head(10)
train['gross'].fillna(train['gross'].dropna().median(), inplace=True)
train['num_critic_for_reviews'].fillna(train['num_critic_for_reviews'].dropna().mean(), inplace=True)
train['duration'].fillna(train['duration'].dropna().mean(), inplace=True)
train['director_facebook_likes'].fillna(train['director_facebook_likes'].dropna().mean(), inplace=True)
train['actor_3_facebook_likes'].fillna(train['actor_3_facebook_likes'].dropna().mean(), inplace=True)
train['actor_1_facebook_likes'].fillna(train['actor_1_facebook_likes'].dropna().mean(), inplace=True)
train['num_voted_users'].fillna(train['num_voted_users'].dropna().mean(), inplace=True)
train['cast_total_facebook_likes'].fillna(train['cast_total_facebook_likes'].dropna().mean(), inplace=True)
train['facenumber_in_poster'].fillna(train['facenumber_in_poster'].dropna().mean(), inplace=True)
train['num_user_for_reviews'].fillna(train['num_user_for_reviews'].dropna().mean(), inplace=True)
train['budget'].fillna(train['budget'].dropna().mean(), inplace=True)
train['actor_2_facebook_likes'].fillna(train['actor_2_facebook_likes'].dropna().mean(), inplace=True)
train['imdb_score'].fillna(train['imdb_score'].dropna().mean(), inplace=True)
train['aspect_ratio'].fillna(train['aspect_ratio'].dropna().mean(), inplace=True)
train['movie_facebook_likes'].fillna(train['movie_facebook_likes'].dropna().mean(), inplace=True)
train.head(10)
train=train.dropna()
X=train.drop('imdb_score',axis=1)
y=train['imdb_score']

train.head()
X.columns
X = X[['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'budget', 'title_year', 'actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X.columns
import xgboost as xgb
from sklearn import cross_validation, metrics
xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)
error_xgb = metrics.mean_squared_error(y_test, predictions_xgb)
from math import sqrt
print(sqrt(error_xgb))
import matplotlib.pylab as plt
import matplotlib.pyplot as plote
f,ax = plote.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
temp = pd.concat([X_train, y], axis=1)
sns.distplot(y)
sns.pairplot(X_train.iloc[:,0:8],palette='inferno')

train[['director_name', 'imdb_score']].groupby(['director_name'], as_index=False).mean().sort_values(by='imdb_score', ascending=False)
train[['director_name','director_facebook_likes', 'imdb_score']].groupby(['director_name'], as_index=False).mean().sort_values(by='imdb_score', ascending=False)
train[['actor_2_name', 'imdb_score']].groupby(['actor_2_name'], as_index=False).mean().sort_values(by='imdb_score', ascending=False)
train[['gross', 'imdb_score']].groupby(['gross'], as_index=False).mean().sort_values(by='imdb_score', ascending=False)
train[['num_critic_for_reviews', 'imdb_score']].groupby(['num_critic_for_reviews'], as_index=False).mean().sort_values(by='imdb_score', ascending=False)