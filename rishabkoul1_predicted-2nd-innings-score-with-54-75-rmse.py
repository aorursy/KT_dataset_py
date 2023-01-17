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
df=pd.read_csv('/kaggle/input/odi-matches-first-innings-scores/matches.csv')
df.head()
df.drop(['innings2_overs','innings2_wickets'],axis=1,inplace=True)
df.head()
df['team2'].value_counts()
df['team2'].unique()
consistent_teams = ['ZIMBABWE', 'SOUTH AFRICA', 'NEW ZEALAND', 'INDIA',
       'SRI LANKA', 'ENGLAND', 'WEST INDIES', 'KENYA', 'AUSTRALIA',
       'PAKISTAN', 'BANGLADESH', 'SCOTLAND', 'AFGHANISTAN',
        'IRELAND', 
        'NETHERLANDS']
df = df[(df['team1'].isin(consistent_teams)) & (df['team2'].isin(consistent_teams))]
df.head()

# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['team1', 'team2'],drop_first=True)
encoded_df.head()
encoded_df.columns
# Rearranging the columns
encoded_df = encoded_df[['innings1_overs', 'innings1_wickets', 'innings1_runs',
       'year', 'team1_AUSTRALIA', 'team1_BANGLADESH', 'team1_ENGLAND',
       'team1_INDIA', 'team1_IRELAND', 'team1_KENYA', 'team1_NETHERLANDS',
       'team1_NEW ZEALAND', 'team1_PAKISTAN', 'team1_SCOTLAND',
       'team1_SOUTH AFRICA', 'team1_SRI LANKA', 'team1_WEST INDIES',
       'team1_ZIMBABWE', 'team2_AUSTRALIA', 'team2_BANGLADESH',
       'team2_ENGLAND', 'team2_INDIA', 'team2_IRELAND', 'team2_KENYA',
       'team2_NETHERLANDS', 'team2_NEW ZEALAND', 'team2_PAKISTAN',
       'team2_SCOTLAND', 'team2_SOUTH AFRICA', 'team2_SRI LANKA',
       'team2_WEST INDIES', 'team2_ZIMBABWE','innings2_runs']]
encoded_df.head()
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='innings2_runs', axis=1)[encoded_df['year'] <= 2016]
X_test = encoded_df.drop(labels='innings2_runs', axis=1)[encoded_df['year'] >= 2017]
y_train = encoded_df[encoded_df['year'] <= 2016]['innings2_runs'].values
y_test = encoded_df[encoded_df['year'] >= 2017]['innings2_runs'].values
# Removing the 'date' column
X_train.drop(labels='year', axis=True, inplace=True)
X_test.drop(labels='year', axis=True, inplace=True)

# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
# Creating a pickle file for the classifier
import pickle
filename = 'second-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
prediction=ridge_regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test,predictions)
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
prediction=lasso_regressor.predict(X_test)
sns.distplot(y_test-prediction)
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
from sklearn.model_selection import RandomizedSearchCV
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)
predictions
import seaborn as sns
sns.distplot(y_test-predictions)
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))