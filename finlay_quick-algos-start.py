%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error



import matplotlib

matplotlib.rcParams['agg.path.chunksize'] = 10000



import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 50)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_test = pd.concat((train, test)).reset_index(drop=True)

train_test = train_test[train.columns]



features = [x for x in train.columns if x not in ['id','loss']]

cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

bin_features = [x for x in cat_features if len(train[x].unique()) == 2]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]
cats = [feat for feat in features if 'cat' in feat]

for feat in cats:

    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]



ntrain = train.shape[0]

train_set = train_test.iloc[:ntrain,:]

test_set = train_test.iloc[ntrain:,:-1]



train_X = train_set.drop(['id', 'loss'], axis = 1)

train_Y = np.log1p(train_set['loss'])
from sklearn import cross_validation

X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(train_X, train_Y, test_size = 0.1, random_state = 520)
from sklearn.linear_model import LinearRegression



model = LinearRegression(n_jobs=-1)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('LinReg MAE:', result)

# LinReg MAE: 1282.78422684
from sklearn.linear_model import Ridge



model = Ridge(alpha = 0.01, random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('Ridge MAE:', result)

# Ridge MAE: 1282.76855425
from sklearn.linear_model import Lasso



model = Lasso(alpha = 0.001,random_state=520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('Lasso MAE:', result)

# Lasso MAE: 1284.21625523
from sklearn.linear_model import ElasticNet



model = ElasticNet(alpha = 0.001,random_state=520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('ElasticNet MAE:', result)

# ElasticNet MAE: 1283.29871562
from sklearn.neighbors import KNeighborsRegressor



model = KNeighborsRegressor(n_neighbors = 5,n_jobs=-1)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('KNN MAE:', result)

# KNN MAE: 1615.10061382
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor(max_depth = 10,random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('CART MAE:', result)

# CART MAE: 1288.19610559
from sklearn.svm import SVR



model = SVR(C = 10)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('SVM MAE:', result)

# too slow...
from sklearn.ensemble import BaggingRegressor



model = BaggingRegressor(n_jobs = -1,n_estimators = 50)

# # model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('Bagging MAE:', result)

# Bagging MAE: 1210.76283562
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_jobs=-1,n_estimators = 50,random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('RF MAE:', result)

# RF MAE: 1211.18864912
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor(n_jobs=-1,n_estimators = 50, random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('ExtraTree MAE:', result)

# ExtraTree MAE: 1214.07295142
from sklearn.ensemble import AdaBoostRegressor



model = AdaBoostRegressor(n_estimators = 50, random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('AdaBoosting MAE:', result)

# AdaBoosting MAE: 1623.08514721
from sklearn.ensemble import GradientBoostingRegressor



model = GradientBoostingRegressor(n_estimators = 50, random_state = 520)

# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('GradientBoosting MAE:', result)

# GradientBoosting MAE: 1269.84382941
from xgboost import XGBRegressor



# model = XGBRegressor(n_estimators = 200, subsample = 0.7 , seed = 0 , colsample_bytree = 0.7 , 

#                     silent = 1, objective = 'reg:linear' , learning_rate = 0.075 , 

#                     max_depth = 6, min_child_weight = 1)



# model.fit(X_train, Y_train)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val)))

# print ('XGB MAE:', result)

# XGB MAE: 1140.69633297
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential

from keras.layers import Dense



# define baseline model

def baseline():

    # create model

    model = Sequential()

    model.add(Dense(200, input_dim = X_train.shape[1], init = 'he_normal'))

    model.add(Dense(1, init = 'he_normal'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model





# model = KerasRegressor(build_fn = baseline, nb_epoch = 20, verbose = 2)

# model.fit(X_train.values, Y_train.values)

# result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val.values)))

# print ('MLP MAE:', result)

# MLP MAE: 1317.24074992