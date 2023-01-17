# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.preprocessing import LabelEncoder
from math import sqrt
%matplotlib inline
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



train.head(5)
test.head(5)
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.head(5)
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
ntrain = train.shape[0]
ntest = test.shape[0]
y = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
len(all_data)
missing_value_df = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False)/len(all_data))
missing_value_df.reset_index(inplace=True)
missing_value_df.columns = ['Feature', 'Missing Value Ratio']
missing_value_df = missing_value_df[missing_value_df['Missing Value Ratio'] > 0.0]
len(missing_value_df)
features_to_be_dropped = list(missing_value_df[missing_value_df['Missing Value Ratio'] >= 0.75].loc[:, 'Feature'].values)
features_to_be_dropped
all_data.shape
all_data.drop(features_to_be_dropped, axis=1, inplace=True)
missing_value_df = missing_value_df[~missing_value_df['Feature'].isin(features_to_be_dropped)]

missing_value_df.head()
category_cols = []
numerical_cols = []
for feature in list(missing_value_df.Feature.unique()):
    if(all_data[feature].dtype == np.float64 or all_data[feature].dtype == np.int64):
          numerical_cols.append(feature)
    else:
          category_cols.append(feature)
for feature in numerical_cols:
    all_data[feature].fillna(all_data[feature].mean(), inplace=True)
for feature in category_cols:
    all_data[feature].fillna(all_data[feature].value_counts().index[0], inplace=True)
all_data.isnull().sum().sort_values(ascending=False)
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]
x = train
x_test = test
from sklearn.cross_validation import KFold
eval_size = 0.20
kf = KFold(len(y), round(1./eval_size))
train_indices, valid_indices = next(iter(kf))

x_train, y_train = x.loc[train_indices], y[train_indices]
x_valid, y_valid = x.loc[valid_indices], y[valid_indices]
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
regressors = [
    LinearRegression(),
    RidgeCV(), 
    LassoCV(), 
    ElasticNetCV(),
    SVR(),
    KNeighborsRegressor(),
    MLPRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    XGBRegressor()]
# Logging for Visual Comparison
log_cols=["Regressor", "Training Accuracy", "Validation Accuracy"]
log = pd.DataFrame(columns=log_cols)
for reg in regressors:
    print("="*30)
    
    reg_name = reg.__class__.__name__
    print(reg_name)

    reg.fit(x_train, y_train)
    
    #Training Accuracy
    y_train_pred = reg.predict(x_train)
    train_acc = mean_squared_error(y_train, y_train_pred)
    
    #Validation Accuracy
    y_valid_pred = reg.predict(x_valid)
    valid_acc = mean_squared_error(y_valid, y_valid_pred)
    
    print("Validation Accuracy: {}".format(valid_acc))
    
    log_entry = pd.DataFrame([[reg_name, train_acc, valid_acc]], columns=log_cols)
    log = log.append(log_entry)
log = log[log.Regressor != 'LinearRegression']
log.sort_values('Validation Accuracy', ascending=False).plot.barh(x='Regressor', y='Validation Accuracy', figsize=(16,7))
from sklearn.model_selection import GridSearchCV
params_grid_1 = {'n_estimators': [50, 110, 200, 300], \
          'max_depth':[3, 7, 10, 14], \
          'learning_rate':[0.05, 0.1], \
          'gamma':[0, 1, 5]
         }

model_1 = XGBRegressor()
grid_search_m1 = GridSearchCV(estimator=model_1, param_grid=params_grid_1, cv=10)
grid_search_m1.fit(x,y)
params_grid_2 = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [20, 50,100,150],
              }

model_2 = GradientBoostingRegressor()
grid_search_m2 = GridSearchCV(estimator=model_2, param_grid=params_grid_2, cv=10)
grid_search_m2.fit(x,y)
# params_grid_3 = {'alpha': list(x / 10 for x in range(0, 101)),
#                     'fit_intercept': [True,False], 
#                     'normalize' :[False, True],
#                     'gcv_mode': ['eigen', 'auto', 'svd'],
#                     'store_cv_values': [False, True]}

# model_3 = RidgeCV()
# grid_search_m3 = GridSearchCV(estimator=model_3, param_grid=params_grid_3, cv=10, n_jobs=-1)
# grid_search_m3.fit(x,y)
results = pd.DataFrame()
results['xgb_pred'] = grid_search_m1.predict(x_test)
results['gbr_pred'] = grid_search_m2.predict(x_test)
# results['rid_pred'] = grid_search_m3.predict(x_test)

results['final'] = results['xgb_pred']*0.5 + results['gbr_pred']*0.50 #+ results['rid_pred']*0.25
y_predict = results['final']
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = y_predict
sub.to_csv('submission.csv',index=False)