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
df.shape
df.info()
df.describe()
# show all columns
pd.set_option('display.max_columns', None)
df.head(5)
# Remove column - 'url' since the details present in url is already available in 
# columns - 'region', 'county', 'state'
df = df.drop('url', axis=1)
df['region_url'].unique()[:10]
# The county column only has 'nan' in it. Therefore, take the county name from 'region_url' column.
df['county'].unique()
# updating 'county' from 'region_from_url' column.
df['county'] = df['region_url'].str.replace('https://','').str.replace('.craigslist.org','')
# drop column - region_url
df = df.drop('region_url', axis=1)
# finding unique list of manufacturer. There are NULL values.
df['manufacturer'].unique()
model_df = df.loc[df['manufacturer'].isnull(), ['model']]
model_df['model'].unique()[:50]
model_df.shape
df.shape
# update 'manufacturer' to 'other' when its NULL
df.loc[df['manufacturer'].isnull(), ['manufacturer']] = 'other'
df.head(5)
df[df['region'].isnull() == True]
df['region'].unique()[:10]
# removing all records which has price = 0.
df = df[df['price'] != 0]
df['year'].unique()
# There are totally 82 records that have year higher than current year ie) 2020. They have to be removed as well.
df[df['year'] > 2020].shape
df = df[df['year'] < 2020]
# 'model' column has lot of inconsistent data eg). Anything, sequoia limited, 30 YEARS.EXP.
df['model'].unique()
# drop columns - lat, long
df = df.drop(['lat','long'], axis=1)
len(df)
# Identify the no.of missing values in each column and their percentage compared to total.
missing_vals = df.isnull().sum().sort_values(ascending = False)
(missing_vals/len(df))*100
# Removing rows which has less than 5% of NULLs in columns.
df=df.dropna(subset=['model','fuel','transmission','title_status','description'])
df.shape
df.head()
df['cylinders'].unique()
df['type'].unique()
# Using forward fill for the columns - 'paint_color', 'drive', 'cylinders', 'type'
df['type'] = df['type'].fillna(method='ffill')
df['paint_color'] = df['paint_color'].fillna(method='ffill')
df['drive'] = df['drive'].fillna(method='ffill')
df['cylinders'] = df['cylinders'].fillna(method='ffill')
df.isnull().sum()
df['condition'].unique()
# updating the condition as 'new' for all vehicles whose year is 2019 and above
df.loc[df.year>=2019, 'condition'] = df.loc[df.year>=2019, 'condition'].fillna('new')
df.groupby(['condition']).count()['year']
df.isnull().sum()
# Addressing the NULLs in 'odometer' column.

# Since odometer is related to the condition of the vehicle, it can be used to fill the missing odometer values.
# The mean of odometer values for each condition is calculated and is used to fill the NULL values for those 
# corresponding condition.
# Find the total distinct values for 'condition'
conditions = list(df['condition'].unique())
conditions.pop(3) # removing null value from list
conditions
# Find the corresponding mean value of 'odometer' for each value in 'condition'
mean_odometer_per_condition_df = df.groupby('condition').mean()['odometer'].reset_index()
mean_odometer_per_condition_df
excellent_odo_mean = df[df['condition'] == 'excellent']['odometer'].mean()
good_odo_mean = df[df['condition'] == 'good']['odometer'].mean()
like_new_odo_mean = df[df['condition'] == 'like new']['odometer'].mean()
salvage_odo_mean = df[df['condition'] == 'salvage']['odometer'].mean()
fair_odo_mean = df[df['condition'] == 'fair']['odometer'].mean()
print('Like new average odometer:', round( like_new_odo_mean,2))
print('Excellent average odometer:', round( excellent_odo_mean,2))
print('Good average odometer:', round( good_odo_mean,2))
print('Fair average odometer:', round( fair_odo_mean,2))
print('Salvage average odometer:', round( salvage_odo_mean,2))
# Update the 'condition' based on the average 'odometer' values for each 'condition'

df.loc[df['odometer'] <= like_new_odo_mean, 'condition'] = df.loc[df['odometer'] <= like_new_odo_mean, 'condition'].fillna('like new')

df.loc[df['odometer'] >= fair_odo_mean, 'condition'] = df.loc[df['odometer'] >= fair_odo_mean, 'condition'].fillna('fair')

df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'].fillna('excellent')

df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'] = df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'].fillna('good')

df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'].fillna('salvage')
# 'model' can be related to 'size'.
# checking the sizes for model = 'patriot'
df.loc[df['model'] == 'patriot', 'size'].unique()

#There are more than 1 size for the same model. Therefore, this column is not reliable and has to be removed. 
# The car details can be scrapped from a different source and then combined to populate the correct car features.
# dropping the column - 'size' since its not reliable
# dropping the column - 'id' since it doesn't have any meaning
# dropping the column - 'image_url' since it doesn't have any meaning
# dropping the column - 'vin' since it doesn't have any meaning
# dropping the column - 'description' - few rows contain important details. dropping for now.

df = df.drop(['size','id','image_url','vin','description'], axis = 1)
import matplotlib.pyplot as plt
sns.countplot(x='condition', data=df, palette=("Paired"))
plt.title('Number of vehicles listed on craigslist across different conditions', fontsize=22)
region_count  = df['region'].value_counts()
region_count = region_count[:10,]
plt.figure(figsize=(11,8))
sns.barplot(region_count.values, region_count.index, alpha=1,palette=("Paired"))
plt.title('Top 10 Counties which has the highest cars listings on Craigslist', fontsize=22)
plt.xlabel('Number of Cars', size="20")
plt.ylabel('County Names', size="20")
plt.show()
region_count  = df['region'].value_counts()
region_count = region_count[-10:,]
plt.figure(figsize=(11,8))
sns.barplot(region_count.values, region_count.index, alpha=1,palette=("Paired"))
plt.title('Top 10 Counties which has the lowest cars listings on Craigslist', fontsize=22)
plt.xlabel('Number of Cars', size="20")
plt.ylabel('County Names', size="20")
plt.show()
state_count  = df['state'].value_counts()
state_count = state_count[-10:,]
plt.figure(figsize=(11,8))
state_count.plot(kind='pie', subplots=True, figsize=(8, 8))
plt.title('Top 10 States which has the highest cars listings on Craigslist', fontsize=22)
plt.xlabel('Number of Cars', size="20")
plt.ylabel('States Abbrevation', size="20")
plt.show()
top_priced_counties = df.groupby('county').sum()['price'].reset_index().sort_values('price', ascending=False)[:10]
top_priced_counties
plt.figure(figsize=(11,8))
sns.barplot(top_priced_counties.price, top_priced_counties.county, alpha=1,palette=("Paired"))
plt.title('Top 10 Counties w.r.t total car price on Craigslist', fontsize=22)
plt.xlabel('Total value of Cars in thousand million (1e9)', size="20")
plt.ylabel('County Names', size="20")
plt.show()
manufacturer_count = df['manufacturer'].value_counts().iloc[:10]
plt.figure(figsize=(10,6))
manufacturer_count.plot(kind='pie', subplots=True, figsize=(8, 8))
plt.title('Top 10 car manufacturer listings on Craigslist', fontsize=22)
plt.xlabel('No.of Listings', size="20")
plt.ylabel('Manufacturer', size="20")
plt.show()
df.info()
df['cylinders'].unique()
df['transmission'].unique()
df['title_status'].unique()
# Removing region since we already have 'county'
df=df.drop('region', axis=1)
# Removing rows which has NULLs in conditon and odometer.
df=df.dropna(subset=['odometer','condition'])
# Identify the no.of missing values in each column and their percentage compared to total.
missing_vals = df.isnull().sum().sort_values(ascending = False)
(missing_vals/len(df))*100
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# convert characters to numbers using label Encoding
df[['county','manufacturer', 'model', 'condition','cylinders', 'fuel', 'title_status', 'transmission','drive', 'type', 'paint_color', 'state']] = df[['county','manufacturer', 'model', 'condition','cylinders', 'fuel', 'title_status', 'transmission','drive','type', 'paint_color', 'state']].apply(le.fit_transform)
df
df["odometer"] = np.sqrt(preprocessing.minmax_scale(df["odometer"]))
df
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
