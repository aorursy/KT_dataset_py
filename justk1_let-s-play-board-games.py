# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
games = pd.read_csv('../input/games.csv')
games.head()
print (games.shape)
# Target variable is Average Rating. So let us explore its variation.
plt.hist(games['average_rating'])
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
games.describe()
# For a game with zero average rating, there are no users rated. 
games[games['average_rating']==0].iloc[0]
# For a game with average rating >0, user have rated game. 
games[games['average_rating']>0].iloc[0]
games_cleaned = games[games['users_rated']>0]
games_cleaned = games_cleaned[games_cleaned['yearpublished']>1900]
games_cleaned = games_cleaned[games_cleaned['maxplaytime']<500]

# Drop any row with no data
games_cleaned = games_cleaned.dropna(axis=0)
games_cleaned.describe()
print ("Percentage data filtered {:.2f}".format((games.shape[0]-games_cleaned.shape[0])/games.shape[0]*100)) # This is quite significant. 
games_cleaned.corr()["average_rating"]
sns.pairplot(games_cleaned[['total_comments','yearpublished','playingtime','total_owners','minage','average_rating']])
df_year_mean  = games_cleaned.groupby(['yearpublished']).mean()
df_year_mean.reset_index(inplace=True)
df_year_mean.head()
df_year_mean.plot.scatter('yearpublished','average_rating',alpha=0.6,sizes=(10, 100))
plt.xlabel('Year Published')
plt.ylabel('Mean Rating')
y = games_cleaned['average_rating']
X = games_cleaned.drop(['average_rating','type','name','id','bayes_average_rating'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training set has {} samples.".format(X_train.shape[0]))
print("Validation set has {} samples.".format(X_val.shape[0]))
# Linear Regression
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LinModel = LinearRegression()
LinModel.fit(X_train,y_train)
y_pred = LinModel.predict(X_val)

error = math.sqrt(mean_squared_error(y_pred,y_val))
print (f'Root mean squared error is {error}')

print ('Coefficients of the linear model are',LinModel.coef_)
print('Intercept of the model is',LinModel.intercept_)
def report_coef(names,coef,intercept):
    r = pd.DataFrame( { 'coef': coef, 'positive': coef>=0  }, index = names )
    r = r.sort_values(by=['coef'])
    display(r)
    print("Intercept: {}".format(intercept))
    r['coef'].plot(kind='barh', color=r['positive'].map({True: 'b', False: 'r'}))
    
column_names = X_train.columns.tolist()
report_coef(
  column_names,
  LinModel.coef_,
  LinModel.intercept_)
from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
RF_model.fit(X_train, y_train)
RF_predictions = RF_model.predict(X_val)
mean_squared_error(RF_predictions, y_val)
import lightgbm as lgb
train_data = lgb.Dataset(X_train,label=y_train)
test_data = lgb.Dataset(X_val,label=y_val)
param = {'num_leaves':131, 'num_trees':100, 'objective':'regression','metric':'rmse'}
num_round = 1000
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])