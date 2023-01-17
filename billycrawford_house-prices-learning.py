import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn import metrics
from sklearn import kernel_ridge
from sklearn import ensemble
from sklearn import neighbors
from tensorflow import keras
import tensorflow as tf
sns.set(style="darkgrid")
all_data = pd.read_csv('../input/house-prices-data-preparation/all_data.csv')
all_data.head(9)
train_df = all_data.loc[~all_data['SalePrice'].isna()]
test_df = all_data.loc[all_data['SalePrice'].isna()].drop(['SalePrice'], axis = 1)


train, val = ms.train_test_split(train_df, test_size=0.2)

X_train = np.array(train.drop(['SalePrice'], axis = 1))
X_val = np.array(val.drop(['SalePrice'], axis = 1))
X_test = np.array(test_df)

Y_train = np.array(train['SalePrice'])
Y_val = np.array(val['SalePrice'])
all_data.loc[all_data['LotFrontage'].isna()]
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.RidgeCV()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = kernel_ridge.KernelRidge()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.Lasso()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.LassoCV()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = ensemble.RandomForestRegressor(n_estimators = 50)
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
param_grid = {
                 'n_estimators': [5, 10, 15, 20, 50],
                 'max_depth': [2, 5, 7, 9]
             }

model = ensemble.RandomForestRegressor()
grid_search = ms.GridSearchCV(model,param_grid, cv = 5)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
param_grid = {'n_neighbors': [1, 5, 10, 25]}

model = neighbors.KNeighborsRegressor()
grid_search = ms.GridSearchCV(model,param_grid, cv = 5)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
param_grid = {'n_neighbors': [1, 5, 10, 25]}

model = neighbors.KNeighborsRegressor(weights='distance')
grid_search = ms.GridSearchCV(model,param_grid, cv = 5)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
NN_model = keras.Sequential([
    keras.layers.Dense(327, activation=tf.nn.relu),
    keras.layers.Dense(1000, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(250, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

NN_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

NN_model.fit(X_train, Y_train, epochs=30)

np.sqrt(metrics.mean_squared_error(Y_val,NN_model.predict(X_val)))
model = linear_model.RidgeCV()
model.fit(X_train,Y_train)


errors = (Y_train - model.predict(X_train)).reshape(-1, 1)

model2 = ensemble.RandomForestRegressor(n_estimators = 50)
model2.fit(errors,Y_train)

# validation
np.sqrt(metrics.mean_squared_error(Y_val,model2.predict(model.predict(X_val).reshape(-1, 1))))


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
param_grid = {'n_estimators': [100, 200, 500],
             'max_depth': [2,4],
             'learning_rate': [0.001,0.01,0.1]}

model = ensemble.GradientBoostingRegressor()
grid_search = ms.GridSearchCV(model,param_grid, cv = 5, verbose = 2,n_jobs = -1)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
grid_search.best_estimator_
param_grid = {'n_estimators': [150, 200, 250],
             'max_depth': [4,6],
             'learning_rate': [0.05,0.1,0.15]}

model = ensemble.GradientBoostingRegressor()
grid_search = ms.GridSearchCV(model,param_grid, cv = 5, verbose = 2,n_jobs = -1)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
grid_search.best_estimator_
param_grid = {'n_estimators': [225, 250, 275],
             'learning_rate': [0.025,0.05,0.075]}

model = ensemble.GradientBoostingRegressor(max_depth = 4)
grid_search = ms.GridSearchCV(model,param_grid, cv = 5, verbose = 2,n_jobs = -1)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
predictions = np.exp(grid_search.predict(X_test))
output = pd.concat([test_df['Id'].reset_index(drop=True), 
                    pd.Series(predictions).reset_index(drop=True)], axis = 1)

output.columns = ['Id','SalePrice']
output.to_csv('output.csv',index=False)
output.head()
