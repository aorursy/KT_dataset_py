from datetime import datetime
print('Process start time :', datetime.now())

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
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
# show all columns
pd.set_option('display.max_columns', None)
df.head(5)
# Removing region since we already have 'county'
df=df.drop(['region', 'region_url', 'vin','url','image_url','description','county'], axis=1)
df=df.dropna()
df.shape
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.columns
df.shape
# convert characters to numbers using label Encoding
df[['size','manufacturer', 'model', 'condition','cylinders', 'fuel', 'title_status', 'transmission','drive', 'type', 'paint_color', 'state']] = df[['size','manufacturer', 'model', 'condition','cylinders', 'fuel', 'title_status', 'transmission','drive','type', 'paint_color', 'state']].apply(le.fit_transform)
df
df["odometer"] = np.sqrt(preprocessing.minmax_scale(df["odometer"]))
# Seperate Features and Outcome
X = df.drop('price',axis=1).values
y = df.price.values
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# works for classification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Create a dataframe to store accuracy scores of different algorithms
accuracy_df = pd.DataFrame(columns=('r2', 'rmse'))
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error as MSE

# Fit
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['Decision Tree Baseline']))
accuracy_df
from sklearn.model_selection import GridSearchCV

scoring = metrics.make_scorer(metrics.mean_squared_error)

param_grid = {
'criterion':['mse'] 
,'splitter':['best','random']
,'max_depth':[4, 5, 6, 7, 8]
,'min_samples_split':[0.8, 2]
,'max_features':['auto','sqrt','log2']
}

g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0),param_grid=param_grid,scoring=scoring, cv=5, refit=True)
g_cv.fit(X_train, y_train)
g_cv.best_params_
result = g_cv.cv_results_
# print(result)

# Predict
y_pred = g_cv.best_estimator_.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['Decision Tree HyperParam Tuning']))
accuracy_df.sort_values('rmse')
from sklearn.ensemble import RandomForestRegressor

# Fit
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['Random Forest Baseline']))
accuracy_df.sort_values('rmse')
from sklearn.ensemble import GradientBoostingRegressor

# Fit
model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['Gradient Boosting Baseline']))
accuracy_df.sort_values('rmse')
from xgboost import XGBRegressor

# Fit
model = XGBRegressor(random_state=0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['XGBoost Baseline']))
accuracy_df.sort_values('rmse')
import xgboost as xgb

model = xgb.XGBRegressor(
#     gamma=1,                 
    learning_rate=0.05,
#     max_depth=3,
#     n_estimators=10000,                                                                    
#     subsample=0.8,
    random_state=34,
    booster='gbtree',    
    objective='reg:squarederror',
    eval_metric='rmse'
) 
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['XGBoost with Parameters']))
accuracy_df.sort_values('rmse')
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()
param_grid = {
#               'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
#               'learning_rate': ['constant'],
#               'learning_rate_init': [0.01],
#               'power_t': [0.5],
#               'alpha': [0.0001],
#               'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]
}
model = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, verbose=True, pre_dispatch='2*n_jobs')

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['MLPRegressor with Parameter Tuning']))
accuracy_df.sort_values('rmse')


#Splitting the training data in to training and validation datasets for Model training

import lightgbm as lgb
from sklearn.model_selection import train_test_split

Xtrain, Xval, Ztrain, Zval = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
#         'max_depth': -1,
#         'subsample': 0.8,
#         'bagging_fraction' : 1,
#         'max_bin' : 5000 ,
#         'bagging_freq': 20,
#         'colsample_bytree': 0.6,
        'metric': 'rmse',
#         'min_split_gain': 0.5,
#         'min_child_weight': 1,
#         'min_child_samples': 10,
#         'scale_pos_weight':1,
#         'zero_as_missing': False,
#         'seed':0,        
    }
model = lgb.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=8000,
                  verbose_eval=500, valid_sets=valid_set)


y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['LightGBM with Parameters']))
accuracy_df.sort_values('rmse')
#Splitting the training data in to training and validation datasets for Model training

import lightgbm as lgb1
from sklearn.model_selection import train_test_split

Xtrain, Xval, Ztrain, Zval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Define categorical features, training and validation data
categorical_positions = []
cat = ['manufacturer','model','condition','cylinders','fuel','odometer','title_status','transmission','drive','type','paint_color','county','state']
for c, col in enumerate(df.columns):
    for x in cat:
        if col == x:
            categorical_positions.append(c-1)


train_set = lgb1.Dataset(Xtrain, label=Ztrain, categorical_feature=categorical_positions)
valid_set = lgb1.Dataset(Xval, label=Zval)

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
#         'max_depth': -1,
#         'subsample': 0.8,
#         'bagging_fraction' : 1,
#         'max_bin' : 5000 ,
#         'bagging_freq': 20,
#         'colsample_bytree': 0.6,
        'metric': 'rmse',
#         'min_split_gain': 0.5,
#         'min_child_weight': 1,
#         'min_child_samples': 10,
#         'scale_pos_weight':1,
#         'zero_as_missing': False,
#         'seed':0,        
    }
model = lgb1.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=8000,
                  verbose_eval=500, valid_sets=valid_set)


y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['LightGBM with Categories & Parameters']))
accuracy_df.sort_values('rmse')
from catboost import CatBoostRegressor, Pool
    
from sklearn.model_selection import train_test_split

Xtrain, Xval, Ztrain, Zval = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

train_set = lgb.Dataset(Xtrain, Ztrain)
valid_set = lgb.Dataset(Xval, Zval)

model = CatBoostRegressor()

model.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], verbose=100, early_stopping_rounds=1000)

y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['CatBoost Baseline']))
accuracy_df.sort_values('rmse')
from catboost import CatBoostRegressor, Pool
    
from sklearn.model_selection import train_test_split

Xtrain, Xval, Ztrain, Zval = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


        
model = CatBoostRegressor(
                          iterations=1000, 
                          depth=8, 
                          learning_rate=0.01, 
                          loss_function='RMSE', 
                          eval_metric='RMSE', 
                          use_best_model=True)

model.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], verbose=100, early_stopping_rounds=1000)

y_pred = model.predict(X_test)

# Metrics
r2 = round(metrics.r2_score(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
# accuracy_df = accuracy_df.drop('CatBoost Parameter Tuning')
accuracy_df = accuracy_df.append(pd.DataFrame({'r2':[r2],'rmse':[rmse]}, index = ['CatBoost Parameter Tuning']))
accuracy_df.sort_values('rmse')
# Plot
plt.figure(figsize=[25,6])
plt.tick_params(labelsize=14)
plt.plot(accuracy_df.index, accuracy_df['rmse'], label = 'RMSE Scores')
plt.legend()
plt.title('RMSE Score comparison for 10 popular models for test dataset')
plt.xlabel('Models')
plt.ylabel('RMSE Scores')
plt.xticks(accuracy_df.index, rotation='vertical')
plt.savefig('graph.png')
plt.show()
print('Process start time :', datetime.now())
# Rerunning MLP Neural Network to save the model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


mlp = MLPRegressor()
param_grid = {
#               'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
#               'learning_rate': ['constant'],
#               'learning_rate_init': [0.01],
#               'power_t': [0.5],
#               'alpha': [0.0001],
#               'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]
}
model = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, pre_dispatch='2*n_jobs')

model.fit(X_train, y_train)
# Save the neural network model
from joblib import dump, load

filename = 'mlp_neural_network_001.joblib'
with open(filename, 'wb') as file:  
    dump(model, file)
# Predict
y_pred = model.predict(X_test)
df1 = pd.DataFrame({"y":y_test,"y_pred":y_pred })
df1.head(50)
