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
import pandas as pd

import numpy as np
data = pd.read_csv('../input/train.csv')

data.head()
X=data.iloc[:,0:11]

X.head()
Y=data['AveragePrice']

Y.head()
test = pd.read_csv('../input/test.csv')

test.head()
X_test = test

X_test.drop('Total Bags',axis=1,inplace=True)

print(X.shape)

print(X_test.shape)

print(Y.shape)
X_test.head()
X_test.shape
X.drop('Total Bags',axis=1,inplace=True)

X.shape
print(X.head())

print(X_test.head())

print(Y.head())
# from sklearn.ensemble import RandomForestRegressor

# rfr = RandomForestRegressor()

# rfr.fit(X,Y)

# pred10 = rfr.predict(X_test)

# trial10 = pd.DataFrame(pred10, columns=['AveragePrice']).to_csv('trial10.csv')
## trial 11

## hyperparameter optimisation maybe



# from sklearn.model_selection import GridSearchCV



# param_grid = {

#     'bootstrap':[True],

#     'max_depth': [5, 15, 25, 35],

#     'max_features': [2, 3],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [100, 300, 500, 700]

# }



# rfr11 = RandomForestRegressor()

# grid_search = GridSearchCV(estimator=rfr11,param_grid=param_grid,

#                           cv=5,n_jobs=4,verbose=2)



# grid_search.fit(X,Y)



# grid_search.best_params_



# best_grid = grid_search.best_estimator_

# pred11 = best_grid.predict(X_test)



# trial11 = pd.DataFrame(pred11, columns=['AveragePrice']).to_csv('trial11.csv')
## trial 12

##

## something to do with k fold learning



# from sklearn.model_selection import cross_val_predict, GridSearchCV

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.preprocessing import MinMaxScaler



# from sklearn.model_selection import KFold

# from sklearn.metrics import mean_squared_error

# from sklearn.metrics import fbeta_score, make_scorer

# from sklearn import metrics

# from sklearn.model_selection import cross_val_score



# k_fold = KFold(n_splits=5,shuffle=True,random_state=0)

# rfrk = RandomForestRegressor()

# mse = metrics.make_scorer(metrics.mean_squared_error)

# print(cross_val_score(rfrk,X,Y,cv=k_fold,n_jobs=4,scoring=mse))



# from sklearn.model_selection import RandomizedSearchCV



# n_estimators = [int(x) for x in np.linspace(start=500, stop=1000, num=5)]



# max_features = ['auto','sqrt']



# max_depth = [int(x) for x in np.linspace(5,50,num=5)]

# max_depth.append(None)



# min_samples_split = [2,5,10]



# min_samples_leaf = [3,6,9]



# bootstrap = [True,False]



# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}



# print(random_grid)



# from sklearn.ensemble import RandomForestRegressor



# rf = RandomForestRegressor()



# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter=100, cv = 5, verbose=2, random_state=88, n_jobs = 6)



# rf_random.fit(X, Y)



# rf_random.best_params_



# best_random = rf_random.best_estimator_

# pred13 = best_random.predict(X_test)



# trial13 = pd.DataFrame(pred13, columns=['AveragePrice']).to_csv('trial13.csv')

## trial 14



# rfr14 = RandomForestRegressor(n_estimators=400,min_samples_split=5,min_samples_leaf=3,max_features='sqrt',max_depth=15,bootstrap=False)

# rfr14.fit(X,Y)

# pred14 = rfr14.predict(X_test)



# trial14 = pd.DataFrame(pred14, columns=['AveragePrice']).to_csv('trial14.csv')
## trial 15

# rfr15 = RandomForestRegressor(n_estimators=1000,min_samples_split=5,min_samples_leaf=2,max_features='sqrt',max_depth=15,bootstrap=False)

# rfr15.fit(X,Y)

# pred15 = rfr15.predict(X_test)



# trial15 = pd.DataFrame(pred15, columns=['AveragePrice']).to_csv('trial15.csv')
## trial 16



from xgboost import XGBRegressor

from xgboost import plot_importance
# XGBRegressor?



# xgbr = XGBRegressor(max_depth=15,learning_rate=0.1,n_estimators=1000,n_jobs=6,)

# xgbr.fit(X,Y,verbose=True)

# pred16 = xgbr.predict(X_test)



# trial16 = pd.DataFrame(pred15, columns=['AveragePrice']).to_csv('trial16.csv')

# xgbr = XGBRegressor(max_depth=20,learning_rate=0.1,n_estimators=1000,n_jobs=6,)

# xgbr.fit(X,Y,verbose=True)

# pred17 = xgbr.predict(X_test)



# trial17 = pd.DataFrame(pred15, columns=['AveragePrice']).to_csv('trial17.csv')

xgbr = XGBRegressor(max_depth=15,learning_rate=0.25,n_estimators=1000,n_jobs=6,)

xgbr.fit(X,Y,verbose=True)

pred18 = xgbr.predict(X_test)
trial18 = pd.DataFrame(pred18, columns=['AveragePrice']).to_csv('trial18.csv')
## PLEASE NOTE THAT THE ID COLUMN HAS BEEN MANUALLY ADDED BY COPYING IT FROM THE TEST DATA CSV

## FOR DOUBLE CHECKING, PLEASE REPLACE THE INDEX COLUMN IN THE PREDICTION CSV WITH THE ID FROM TEST.CSV