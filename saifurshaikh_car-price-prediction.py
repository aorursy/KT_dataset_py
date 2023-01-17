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
train_df = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')
test_df = pd.read_csv('../input/used-cars-price-prediction/test-data.csv')
train_df.head()
test_df.head()
test_df['Price'] = np.nan
test_df = test_df[train_df.columns]
df = pd.concat([train_df, test_df], axis=0)
df.head()
df[df.Price.isnull()].shape
df.info()
df.isnull().sum()
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()
df.Year.describe()
#### We have data until 2019 and we are in 2020 now. 
#### So, created a new column Vehicle_Age
current_year = 2020
df['Vehicle_Age'] = current_year-df.Year
df.drop('Year', axis=1, inplace=True)
df.head()
df['Mileage_kg_l'] = 0
df['Engine_CC'] = 0
df['Power_BHP'] = 0
df.head()
df.Mileage_kg_l = df.Mileage.str.split(expand=True)[0]
df.Engine_CC = df.Engine.str.split(expand=True)[0]
df.Power_BHP = df.Power.str.split(expand=True)[0]
df.head()
df.Power_BHP = pd.to_numeric(df.Power_BHP, errors='coerce')
df.info()
df.Mileage_kg_l = pd.to_numeric(df.Mileage_kg_l, errors='coerce')
df.Engine_CC = pd.to_numeric(df.Engine_CC, errors='coerce')
df.drop(['Engine', 'Mileage', 'Power'], axis=1, inplace=True)
df.head()
df.info()
df.isnull().sum()
df.Seats.value_counts()
df.Name.value_counts()
df.loc[df['Seats'].isnull()]
df.drop('New_Price', axis=1, inplace=True)
df.head()
df.isnull().sum()
for col in ['Mileage_kg_l','Engine_CC','Power_BHP']: 
    df[col] = df[col].fillna(df[col].median(), axis=0)
df.isnull().sum()
df['Seats'].fillna(value=5.0, inplace=True)
df.isnull().sum()
df.Name = df.Name.astype('category')
df['Name_cat'] = df.Name.cat.codes
df.drop('Name', axis=1, inplace=True)
df.head()
df_dummy = pd.get_dummies(data=df, columns=['Location','Fuel_Type','Transmission','Owner_Type'], drop_first=True)
df_dummy.head()
df_dummy.isnull().sum()
train_data = df_dummy[df_dummy.Price>=0]
test_data = df_dummy[df_dummy.Price.isnull()]
del test_data['Price']
print(train_data.shape)
print(test_data.shape)
X = train_data.drop('Price', axis=1)
y = train_data['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse
print("Accuracy on Traing set: ",svr.score(X_train,y_train))
print("Accuracy on Testing set: ",svr.score(X_test,y_test))
from sklearn.tree import ExtraTreeRegressor
et_reg = ExtraTreeRegressor(random_state=2)
et_reg.fit(X_train, y_train)
et_reg.feature_importances_
#plot graph of feature importances for better visualization

import matplotlib.pyplot as plt
plt.figure(figsize = (12,8))
feat_importances = pd.Series(et_reg.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()
feat_importances = pd.Series(et_reg.feature_importances_, index=X.columns)
feat_importances.nlargest(10).index
X_train_new = X_train[['Transmission_Manual', 'Power_BHP', 'Vehicle_Age', 'Fuel_Type_Diesel',
       'Engine_CC', 'Seats', 'Mileage_kg_l', 'Kilometers_Driven', 'Name_cat',
       'Owner_Type_Second']]
X_test_new = X_test[['Transmission_Manual', 'Power_BHP', 'Vehicle_Age', 'Fuel_Type_Diesel',
       'Engine_CC', 'Seats', 'Mileage_kg_l', 'Kilometers_Driven', 'Name_cat',
       'Owner_Type_Second']]
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_new, y_train)
y_pred = rf_reg.predict(X_test_new)
print("Accuracy on Traing set: ",rf_reg.score(X_train_new,y_train))
print("Accuracy on Testing set: ",rf_reg.score(X_test_new,y_test))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(mse)
print(rmse)
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
params = {'n_estimators' : [100, 300, 500, 700, 1000, 1200, 1500],
         'max_features' : ['auto', 'log2', 'sqrt'],
         'max_depth' : [3,4,5,6,7],
         'min_samples_split' : [10, 15, 20, 25], 
         'min_samples_leaf' : [5, 10, 15, 20]}
from sklearn.model_selection import KFold, RandomizedSearchCV
kfold = KFold(n_splits=5, random_state=2, shuffle=False)
regressor = RandomizedSearchCV(rf_reg, param_distributions=params, cv=kfold, scoring='neg_mean_squared_error')
regressor.fit(X_train_new, y_train)
regressor.best_params_
regressor.best_estimator_
reg = RandomForestRegressor(max_depth=7, max_features='log2', min_samples_leaf=5,
                      min_samples_split=25, n_estimators=500)
reg.fit(X_train_new, y_train)
y_pred = reg.predict(X_test_new)
print("Accuracy on Traing set: ",reg.score(X_train_new,y_train))
print("Accuracy on Testing set: ",reg.score(X_test_new,y_test))
print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
# Hyperparameter Tuned RF gives bad scores as compared to the default parameter values
import xgboost as xgb
xgb_reg = xgb.XGBRegressor(random_state=2)
xgb_reg.fit(X_train_new, y_train)
y_pred = xgb_reg.predict(X_test_new)
print("Accuracy on Traing set: ",xgb_reg.score(X_train_new,y_train))
print("Accuracy on Testing set: ",xgb_reg.score(X_test_new,y_test))
print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          "max_depth"        : [3, 4, 5, 6],
          "min_child_weight" : [1, 3, 5, 7],
          "gamma"            : [0.0, 0.1, 0.2, 0.3, 0.4],
          "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]}
kfold = KFold(n_splits=5, random_state=2, shuffle=False)
regressor = RandomizedSearchCV(xgb_reg, param_distributions=params, cv=kfold, scoring='neg_mean_squared_error')
regressor.fit(X_train_new, y_train)
regressor.best_score_
regressor.best_estimator_
regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.2, max_delta_step=0, max_depth=4,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=2,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train_new, y_train)
y_pred = regressor.predict(X_test_new)
print("Accuracy on Traing set: ",regressor.score(X_train_new,y_train))
print("Accuracy on Testing set: ",regressor.score(X_test_new,y_test))
print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
