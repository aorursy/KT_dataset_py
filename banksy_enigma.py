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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as  xgb
from sklearn import metrics
train = pd.read_csv("../input/train_NIR5Yl1.csv")
test = pd.read_csv("../input/test_8i3B3FC.csv")
#submission = pd.read_csv("sample_submission_OR5kZa5.csv")
train.head()
test.head()
le = LabelEncoder()
train['Tag'] = le.fit_transform(train['Tag'])
test['Tag'] = le.transform(test['Tag'])
train["Reputation-s2"] = train["Reputation"] ** 2
train["Reputation-s3"] = train["Reputation"] ** 3
train["Reputation-Sq"] = np.sqrt(train["Reputation"])
train["Views-s2"] = train["Views"] ** 2
train["Views-s3"] = train["Views"] ** 3
train["Views-Sq"] = np.sqrt(train["Views"])
train["Views-log"] = np.power(10 * np.log(train['Views'] ), 2)
train["Answers-s2"] = train["Answers"] ** 2
train["Answers-s3"] = train["Answers"] ** 3
train["Answers-Sq"] = np.sqrt(train["Answers"])

test["Reputation-s2"] = test["Reputation"] ** 2
test["Reputation-s3"] = test["Reputation"] ** 3
test["Reputation-Sq"] = np.sqrt(test["Reputation"])
test["Views-s2"] = test["Views"] ** 2
test["Views-s3"] = test["Views"] ** 3
test["Views-Sq"] = np.sqrt(test["Views"])
test["Views-log"] = np.power(10 * np.log(test['Views'] ), 2)
test["Answers-s2"] = test["Answers"] ** 2
test["Answers-s3"] = test["Answers"] ** 3
test["Answers-Sq"] = np.sqrt(test["Answers"])
y = train[['Upvotes']]
train.drop(['Upvotes'], axis = 1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split( train,
                                               y, test_size=0.25)
#xgb1 = XGBRegressor()
#parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#              'objective':['reg:linear'],
#              'learning_rate': [.03, 0.05, .07], #so called `eta` value
#              'max_depth': [5, 6, 7],
#              'min_child_weight': [4],
#              'silent': [1],
#              'subsample': [0.7],
#              'colsample_bytree': [0.7],
#              'n_estimators': [500, 1000],
#              'gamma' :[i/10.0 for i in range(0,5)]}
#xgb_grid = GridSearchCV(xgb1,
#                        parameters,
#                        cv = 2,
#                        n_jobs = -1,
#                        verbose=10)
#xgb_grid.fit(train,y)
#print(xgb_grid.best_score_)
#print(xgb_grid.best_params_)
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


dtrain = xgb.DMatrix(X_train, label=y_train)
del(X_train)
dvalid = xgb.DMatrix(X_valid)
del(X_valid)

def xgb_evaluate(max_depth, gamma, colsample_bytree, subsample, reg_alpha):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.05,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree,
              'subsample': subsample,
              'reg_alpha': reg_alpha}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (1, 10), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9),
                                             'subsample': (0.2, 0.9),
                                             'reg_alpha': (0, 0.5)})

xgb_bo.maximize()
params = xgb_bo.res['max']['max_params']
params['max_depth'] = int(params['max_depth'])

model2 = xgb.train(params, dtrain, num_boost_round=250)
# Predict on testing and training set
y_pred = model2.predict(dvalid)
y_train_pred = model2.predict(dtrain)

# Report testing and training RMSE
print(np.sqrt(mean_squared_error(y_valid, y_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
dtrain_full = xgb.DMatrix(train, label=y)
dtest = xgb.DMatrix(test)
model_full = xgb.train(params, dtrain_full, num_boost_round=250)
y_pred_test = model2.predict(dtest)
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.regularizers import l1
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#
#import keras
#
## Convert data as np.array
#features = np.array(train)
#targets = np.array(y.values.reshape(y.shape[0],1))
#print(features[:10])
#print(targets[:10])
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#
## Building the model
#model = Sequential()
#model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(Dropout(.2))
#model.add(Dense(16, activation='relu'))
#model.add(Dropout(.1))
#model.add(Dense(1))
#
## Compiling the model
#model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
#model.summary()
## Training the model
#epochs_tot = 100
#epochs_step = 25
#epochs_ratio = int(epochs_tot / epochs_step)
#
#history = model.fit(features, targets, epochs=25, batch_size=100, verbose=1)
#predictions = model.predict(features_validation, verbose=1)
#print('MSE score = ',mean_squared_error(y_, predictions), '/ 0.0')
#predictions_test = model.predict(np.array(test), verbose=1)



submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['Upvotes'] = y_pred_test
submission.to_csv('submission3.csv', index=False)
