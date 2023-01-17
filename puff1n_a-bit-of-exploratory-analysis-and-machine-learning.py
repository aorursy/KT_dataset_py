# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# import pygeohash as gh





import pickle

from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingCVRegressor
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
# Check for nan values

df.isna().sum()
# drop columns with nonzero NaN and ids for simplicity

cols_for_drop = list(df.columns[df.isna().sum() != 0])

cols_for_drop.extend(['id', 'host_id'])

print(f'Dropped columns: {cols_for_drop}')

df = df.drop(labels=cols_for_drop, axis=1)

# df = df.drop('id', axis=1)

# df = df.drop('host_id', axis=1)
df.dtypes
# convert objects fields to category

for colname in df.columns:

    if df[colname].dtype == 'O': 

        df[colname] = df[colname].astype('category')

print(df.dtypes)
cat_cols = df.select_dtypes(include='category').columns

num_cols = df.select_dtypes(exclude='category').columns

print(f'Category columns: {cat_cols}')

print(f'Numeric columns: {num_cols}')
# Heatmap of correlation

sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr())
# We can check the vaiance and distribution of prices



plt.hist(df.price);

print(f'Variance of prices: {df.price.var()}')



# Almost all (>99%) of samples are <800$

print(f'Percentiles: {np.percentile(df.price, [25, 50, 99])}')
# Apply log to price and delete samples with 0 price



df = df[df.price>0]

df['price_log'] = np.log(df.price)
# Add log price to numeric columns list

num_cols = num_cols.append(pd.core.indexes.base.Index(['price_log']))
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.hist(df.price_log, align='mid', bins=np.arange(2, 9.5, .5), density=True); 

plt.title('Log price')

# plt.grid()



plt.subplot(1, 2, 2)

sns.boxplot(x='neighbourhood_group', y='price_log', data=df)

plt.axhline(df['price_log'][df.neighbourhood_group == 'Bronx'].median(), ls='--')
# Drop samples with price > 1000 

df = df[df['price'] < 1000]
# Catplot with colored markers for neighbourhood_group



sns.catplot(x='room_type', y='price_log', data=df, hue='neighbourhood_group', zorder=0)

# plt.scatter([0, 1, 2], 

#             [df.price_log[df.room_type == 'Entire home/apt'].mean(), 

#              df.price_log[df.room_type == 'Private room'].mean(),

#              df.price_log[df.room_type == 'Shared room'].mean()], c='black', marker='*', s=150)

plt.axhline(df.price_log[df.room_type == 'Entire home/apt'].mean(), xmin=-0.5, xmax=.29, ls='--', c='gray')

plt.axhline(df.price_log[df.room_type == 'Private room'].mean(), xmin=0.35, xmax=0.65, ls='--', c='gray')

plt.axhline(df.price_log[df.room_type == 'Shared room'].mean(), xmin=0.75, xmax=.9, ls='--', c='gray')



plt.axhline(df.price_log[df.room_type == 'Entire home/apt'].median(), xmin=-0.5, xmax=.29, ls='--', c='cyan')

plt.axhline(df.price_log[df.room_type == 'Private room'].median(), xmin=0.35, xmax=0.65, ls='--', c='cyan')

plt.axhline(df.price_log[df.room_type == 'Shared room'].median(), xmin=0.75, xmax=.9, ls='--', c='cyan')

plt.text(2.7, 3.23, '--- mean')
# Robust Scaler and OHEncoder



rscaler = RobustScaler()

ohencoder = OneHotEncoder()



# Numeric Features

num_x_scaled = rscaler.fit_transform(df[num_cols])

num_x_scaled = pd.DataFrame(num_x_scaled, columns=num_cols)



# Categorical Features 

cat_x_dumm = ohencoder.fit_transform(df[cat_cols])

cat_x_dumm = cat_x_dumm.toarray()



# For saving feature names 

cat_x_dumm = pd.DataFrame(cat_x_dumm, columns=ohencoder.get_feature_names(cat_cols))



# Make new dataframe

prep_df = pd.concat([num_x_scaled, cat_x_dumm], axis=1)
X = prep_df.drop(['price'], axis=1)

y = prep_df['price'] 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_test.shape)

print(y_test.shape)
%%time



lgb = LGBMRegressor(boosting_type='gbdt')

gs_lgb = RandomizedSearchCV(lgb, param_distributions={'num_leaves' : np.arange(5, 10, 1), 

                                                      'learning_rate' : np.linspace(.01, 0.1, 5), 

                                                      'n_estimators' : np.arange(50, 560, 100)}, 

                            verbose=2, cv=3, n_jobs=-1)

gs_lgb.fit(X_train, y_train)
gs_lgb.best_estimator_
# Save best configuration and print mse on the test sample 

best_lgb = gs_lgb.best_estimator_

pred_lgb = best_lgb.predict(X_test)

mse_lgb = mean_squared_error(y_test, pred_lgb)

print(gs_lgb.best_estimator_)

print(f'MSE: {mse_lgb:.7f}')



# LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

#               importance_type='split', learning_rate=0.05500000000000001,

#               max_depth=-1, min_child_samples=20, min_child_weight=0.001,

#               min_split_gain=0.0, n_estimators=350, n_jobs=-1, num_leaves=9,

#               objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,

#               silent=True, subsample=1.0, subsample_for_bin=200000,

#               subsample_freq=0)
plt.scatter(pred_lgb, y_test)
# xgb = XGBRegressor()

# params_xgb = {

#     'max_depth' : np.arange(1, 17, 4),

#     'learning_rate' : np.arange(0.01, .15, 0.05),

#     'n_estimators' : np.arange(50, 150, 50),

#     'gamma' : [0, 1, 2]

# }

# gs_xgb = RandomizedSearchCV(xgb, params_xgb, n_jobs=-1, verbose=2)

# gs_xgb.fit(X_train, y_train)
# best_xgb = gs_xgb.best_estimator_

# mse_xgb = mean_squared_error(y_test, best_xgb.predict(X_test))

# print(gs_xgb.best_estimator_)

# print(f'MSE: {mse_xgb:.3f}')
# lr = LinearRegression(n_jobs=-1)

# lr.fit(X_train, y_train)



# mse_lr = mean_squared_error(y_test, lr.predict(X_test))

# print(f'MSE: {mse_lr:.3f}')
# # LASSO

# las = Lasso()

# params_lasso = {

#     'alpha' : np.arange(0.0001, 0.02, .0005)

# }

# gs_lasso = RandomizedSearchCV(las, n_jobs=4, cv=3, param_grid=params_lasso, verbose=5)

# gs_lasso.fit(X_train, y_train)



# best_lasso = gs_lasso.best_estimator_

# mse_lasso = mean_squared_error(y_test, best_lasso.predict(X_test))

# print(gs_lasso.best_estimator_)

# print(f'mse {mse_lasso:.3f}')
# # Ridge



# rid = Ridge()

# params_ridge = {

#     'alpha' : np.linspace(0.1, 1.5, 14)

# }

# gs_rid = RandomizedSearchCV(rid, n_jobs=-1, cv=3, param_grid=params_ridge, verbose=5)

# gs_rid.fit(X_train, y_train)



# best_rid = gs_rid.best_estimator_

# mse_rid = mean_squared_error(y_test, best_rid.predict(X_test))

# print(gs_rid.best_estimator_)

# print(f'{mse_rid:.3f}')
# # Elastic NET







# ela = ElasticNet()

# params_ela = {

#     'alpha' : np.linspace(0.1, 1.5, 14), 

#     'l1_ratio' : np.arange(0.1, 1.5, .1)

# }

# gs_ela = RandomizedSearchCV(ela, n_jobs=4, cv=3, param_grid=params_ela, verbose=5)

# gs_ela.fit(X_train, y_train)





# best_ela = gs_ela.best_estimator_

# mse_ela = mean_squared_error(y_test, best_ela.predict(X_test))

# print(gs_ela.best_estimator_)

# print(f'{mse_ela:.3f}')
# knn = KNeighborsRegressor(algorithm='brute')



# params_knn = {

#     'n_neighbors' : np.arange(3, 51, 6),

#     'leaf_size' : np.arange(2, 20, 2), 

#     'weights' : ['uniform', 'distance']

# }

# gs_knn = RandomizedSearchCV(KNeighborsRegressor(), params_knn, cv=3, verbose=15, n_jobs=-1)

# gs_knn.fit(X_train, y_train)

# best_knn = gs_knn.best_estimator_

# mse_knn = mean_squared_error(y_test, best_knn.predict(X_test))

# print(gs_knn.best_params_)

# print(f'{mse_knn:.3f}')





# ##

# #'leaf_size': 2, 'n_neighbors': 33, 'weights': 'distance'
# plt.bar(['mse_lasso', 'mse_rid', 'mse_ela', 'mse_knn'], [mse_lasso, mse_rid, mse_ela, mse_knn])

# plt.title('MSE')