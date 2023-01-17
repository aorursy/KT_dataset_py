#Imports



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network  import MLPRegressor

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score
import xgboost as xgb

from xgboost import XGBRegressor
#Loading and glancing at data statistics

train_df = pd.read_csv('./firstoffour/train.csv')

test_df = pd.read_csv('./firstoffour/test.csv')
train_df.head()
train_df.info()
train_df.describe()
X, Y = train_df.drop(['AveragePrice'], axis = 1), train_df['AveragePrice']



xgb = XGBRegressor(n_estimators=8000, n_jobs= -1)

xgb.fit(X, Y)

    

test = test_df



test_ID = test_df['id']





pred = xgb.predict(test).reshape((-1,1))

ans = pd.DataFrame(pred, test_ID, columns = ['AveragePrice'])

ans.to_csv('ans.csv', index_label = ['id'])
scorer = make_scorer(mean_squared_error)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 15000, num = 40)]



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]

max_depth.append(3)

# Minimum number of samples required to split a node

subsample = [0.5, 0.7, 0.9 , 1]

# Minimum number of samples required at each leaf node

colsample_bytree = [0.5, 0.7, 0.85, 1]

#gamma = [0, 1,5,10,100, 1000, 10000]

eta = [0.001, 0.01, 0.02, 0.05, 0.15, 0.3]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'subsample': subsample,

               'colsample_bytree': colsample_bytree, 

               #'gamma' : gamma,

               'eta' : eta

              }



clf = XGBRegressor()



grid_obj = RandomizedSearchCV(estimator = clf, n_iter=400, param_distributions=random_grid, cv = 3, n_jobs= -1, random_state=42, verbose = 2)



grid_fit = grid_obj.fit(X,Y)



best_clf = grid_fit.best_estimator_   



                       

pred = best_clf.predict(test).reshape((-1,1))

ans = pd.DataFrame(pred, test_ID, columns = ['AveragePrice'])

ans.to_csv('ans.csv', index_label = ['id'])
best_clf.get_xgb_params
plt.bar(range(len(best_clf.feature_importances_)), best_clf.feature_importances_)

plt.show()


bclf = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.7, eta=0.01, gamma=0, learning_rate=0.1,

       max_delta_step=0, max_depth=6,min_child_weight=1, missing=None,

       n_estimators=5500, n_jobs=5, nthread=None, objective='reg:linear',

       random_state=6, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

       seed=None, silent=True, subsample=1)



#grid_obj = RandomizedSearchCV(estimator = clf, n_iter=400, param_distributions=random_grid, cv = 3, n_jobs= -1, random_state=42, verbose = 2)



bfit = bclf.fit(X,Y)



#best_clf = grid_fit.best_estimator_   



                       

pred = bclf.predict(test).reshape((-1,1))

ans = pd.DataFrame(pred, test_ID, columns = ['AveragePrice'])

ans.to_csv('ans4.csv', index_label = ['id'])

bclf = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.7, eta=0.01, gamma=0, learning_rate=0.1,

       max_delta_step=0, max_depth=6, min_child_weight=1, missing=None,

       n_estimators=5500, n_jobs=5, nthread=None, objective='reg:linear',

       random_state=5, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

       seed=None, silent=True, subsample=1)



#grid_obj = RandomizedSearchCV(estimator = clf, n_iter=400, param_distributions=random_grid, cv = 3, n_jobs= -1, random_state=42, verbose = 2)



bfit = bclf.fit(X,Y)



#best_clf = grid_fit.best_estimator_   



                       

pred = bclf.predict(test).reshape((-1,1))

ans = pd.DataFrame(pred, test_ID, columns = ['AveragePrice'])

ans.to_csv('ans4.csv', index_label = ['id'])

#Log label experiment
scorer = make_scorer(mean_squared_error)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 5500, stop = 8000, num = 5)]



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(4, 12, num = 8)]

#max_depth.append(3)

# Minimum number of samples required to split a node

subsample = [1]

# Minimum number of samples required at each leaf node

colsample_bytree = [0.7]

gamma = [0, 1,5,10,100, 1000, 10000]

eta = [0.001, 0.01, 0.02, 0.05, 0.15, 0.3]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'subsample': subsample,

               'colsample_bytree': colsample_bytree, 

               #'gamma' : gamma,

               #'eta' : eta

              }



clf = XGBRegressor()



grid_obj = RandomizedSearchCV(estimator = clf, n_iter=50, param_distributions=random_grid, cv = 3, n_jobs= -1, random_state=42, verbose = 2)



grid_fit = grid_obj.fit(X,np.log(Y))



best_clf3 = grid_fit.best_estimator_   



                       

pred = best_clf3.predict(test).reshape((-1,1))

ans = pd.DataFrame(np.exp(pred), test_ID, columns = ['AveragePrice'])

ans.to_csv('ans3.csv', index_label = ['id'])
scorer = make_scorer(mean_squared_error)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 5500, stop = 8000, num = 5)]



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(4, 12, num = 8)]

#max_depth.append(3)

# Minimum number of samples required to split a node

subsample = [1]

# Minimum number of samples required at each leaf node

colsample_bytree = [0.7]

gamma = [0, 1,5,10,100, 1000, 10000]

eta = [0.001, 0.01, 0.02, 0.05, 0.15, 0.3]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'subsample': subsample,

               'colsample_bytree': colsample_bytree, 

               #'gamma' : gamma,

               #'eta' : eta

              }



clf = XGBRegressor()



grid_obj = GridSearchCV(estimator = clf, param_grid=random_grid, cv = 3, n_jobs= -1, verbose = 2)



grid_fit = grid_obj.fit(X,np.log(Y))



best_clf3 = grid_fit.best_estimator_   



                       

pred = best_clf3.predict(test).reshape((-1,1))

ans = pd.DataFrame(np.exp(pred), test_ID, columns = ['AveragePrice'])

ans.to_csv('ans3.csv', index_label = ['id'])
best_clf3.get_xgb_params
scorer = make_scorer(mean_squared_error)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 15000, num = 40)]



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]

max_depth.append(3)

# Minimum number of samples required to split a node

subsample = [0.5, 0.7, 0.9 , 1]

# Minimum number of samples required at each leaf node

colsample_bytree = [0.5, 0.7, 0.85, 1]

gamma = [0, 1,5,10,100, 1000, 10000]

eta = [0.001, 0.01, 0.02, 0.05, 0.15, 0.3]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'subsample': subsample,

               'colsample_bytree': colsample_bytree, 

               'gamma' : gamma,

               'eta' : eta

              }



clf = XGBRegressor()



grid_obj = RandomizedSearchCV(estimator = clf, n_iter=600, param_distributions=random_grid, cv = 3, n_jobs= -1, random_state=42, verbose = 2)



grid_fit = grid_obj.fit(X,Y)



best_clf2 = grid_fit.best_estimator_   



                       

pred = best_clf.predict(test).reshape((-1,1))

ans = pd.DataFrame(pred, test_ID, columns = ['AveragePrice'])

ans.to_csv('ans2.csv', index_label = ['id'])
best_clf2.get_xgb_params