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
from numpy.random import seed

seed(22)



import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

import math

import gc

from scipy.stats.stats import kendalltau

from pylab import rcParams



from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder, RobustScaler, PolynomialFeatures

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn import linear_model

from sklearn.pipeline import Pipeline



from xgboost import XGBRegressor

from xgboost import plot_importance

from lightgbm import LGBMRegressor

from lightgbm import plot_importance as lgb_importance



import datetime

from datetime import datetime
train_data = pd.read_csv('/kaggle/input/predict-number-of-upvotes/train.csv')

test_data = pd.read_csv('/kaggle/input/predict-number-of-upvotes/test.csv')

submission = pd.read_csv('/kaggle/input/predict-number-of-upvotes/sample_submission.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

for col in train_data.columns[~train_data.columns.isin(['tag'])]:

    train_data[col] = train_data[col].apply(lambda x: int(x))

for col in test_data.columns[~test_data.columns.isin(['tag'])]:

    test_data[col] = test_data[col].apply(lambda x: int(x))
test_data.loc[(test_data['id'] == 121644)]
train_data.head(10)
print('Train Data shape: ', train_data.shape)

train_data.describe()
train_data.isnull().sum()
train_data.nunique()
train_data.dtypes
test_data.head()
plt.figure(figsize = (10, 5))

sns.barplot(train_data['tag'], train_data['upvotes'])

plt.title('Upvotes by question category')

plt.show()
plt.figure(figsize = (10, 5))

sns.barplot(train_data['tag'], train_data['views'])

plt.title('Views by question category')

plt.show()
plt.figure(figsize = (10, 5))

sns.lineplot(train_data['answers'], train_data['upvotes'])

sns.despine()
plt.figure(figsize = (16, 10))

sns.scatterplot(x = 'upvotes', y = 'views', data = train_data)

sns.despine()
plt.figure(figsize = (16, 5))

sns.scatterplot(x = 'upvotes', y = 'answers', data = train_data)

sns.despine()
plt.figure(figsize = (16, 5))

sns.scatterplot(x = 'views', y = 'answers', data = train_data)

sns.despine()
#4118

train_data.loc[(train_data['views'] > 3000000)]

train_data.loc[(train_data['upvotes'] > 400000)]
#train_data = train_data.loc[(train_data['upvotes'] < 400000)]

train_data = train_data.loc[(train_data['views'] < 3000000)]
train_data.head()
train_data.hist(figsize = (16, 20), bins = 50, xlabelsize = 8, ylabelsize = 8)
user_rep = (train_data['username'].astype(str) + "_" + train_data['reputation'].astype(str)).unique()

print(len(user_rep))

users = (train_data['username'].astype(str)).unique()

print(len(users))
max(users)
users = train_data.username.unique().tolist() + list(set(train_data.username.unique().tolist()) - set(test_data.username.unique().tolist()))

users = pd.DataFrame({'username': users})
user_counts = pd.DataFrame(train_data.groupby('username')['id'].count()).reset_index()

user_counts.columns = ['username', 'questions_train']

user_counts_test = pd.DataFrame(test_data.groupby('username')['id'].count()).reset_index()

user_counts_test.columns = ['username', 'questions_test']

users = users.merge(user_counts_test, on = 'username', how = 'left')

users = users.merge(user_counts, on = 'username', how = 'left')

users['diff'] = abs(users['questions_test'] - users['questions_train'])

users.sort_values(by = ['diff'], ascending = False)
users.loc[(users['questions_test'] <= 100000) & (users['questions_train'].isnull())]
users.loc[(users['questions_train'] <= 20) & (users['diff'] >= 20)]
train_data['rep_ans_vw_avg'] = (train_data['reputation'] + train_data['answers'] + train_data['views'])/3

test_data['rep_ans_vw_avg'] = (test_data['reputation'] + test_data['answers'] + test_data['views'])/3

train_data['rep_ans_vw_avg'] = train_data['rep_ans_vw_avg'].apply(lambda x: int(x))

test_data['rep_ans_vw_avg'] = test_data['rep_ans_vw_avg'].apply(lambda x: int(x))



print('Train Data shape: ', train_data.shape)

print('Test Data shape: ', test_data.shape)



train_data.head(30)
train_data.loc[(train_data['upvotes'] >= 5000)]
fig, ax = plt.subplots(figsize=(8, 8))  # Sample figsize in inches

sns.heatmap(train_data[train_data.columns[~train_data.columns.isin(['id', 'username'])]].corr(), annot = True, square = True, vmin = -1, vmax = 1)
kendallcorr = train_data.corr(method = 'kendall')

fig, ax = plt.subplots(figsize = (8, 8))  # Sample figsize in inches

sns.heatmap(kendallcorr, xticklabels = kendallcorr.columns.values, yticklabels = kendallcorr.columns.values, cmap = "YlGnBu", annot = True)
le = LabelEncoder()

train_data['tag'], test_data['tag'] = le.fit_transform(train_data['tag']), le.fit_transform(test_data['tag'])



X = train_data[train_data.columns[~train_data.columns.isin(['upvotes','username', 'id'])]]

y = train_data['upvotes']

#X['tag'] = X['tag'].astype('category')

#test_data['tag'] = test_data['tag'].astype('category')

#scale_features = X.columns[~X.columns.isin(['tag'])]
scale_features = X.columns[~X.columns.isin(['tag'])]

ss = StandardScaler()

X = ss.fit_transform(X)

testIDs = test_data['id']

test_data = test_data[test_data.columns[~test_data.columns.isin(['username', 'id'])]]

test_data = ss.fit_transform(test_data)

print(X.shape, test_data.shape)
"""rs = RobustScaler().fit(X)

X = rs.transform(X)

testIDs = test_data['id']

test_data = test_data[test_data.columns[~test_data.columns.isin(['username', 'id'])]]

rs = RobustScaler().fit(test_data)

test_data = rs.transform(test_data)

print(X.shape, test_data.shape)



#y = np.log1p(y)

"""
pd.set_option('display.max_columns', None)

train_data[train_data.columns[~train_data.columns.isin(['upvotes','username', 'id'])]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 22)
poly_reg = PolynomialFeatures(degree = 4, interaction_only = False, include_bias = True)  #using polynomial funtion of degree 4, as there are 4 features in given datasets.

X_poly = poly_reg.fit_transform(X_train)

poly_reg.fit(X_train, y_train)

testData = poly_reg.fit_transform(test_data)
lin_reg = linear_model.LassoLars(alpha = 0.021, max_iter = 200)

lin_reg.fit(X_poly, y_train)

y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))



print('Linear Model RMSE score is', math.sqrt(mean_squared_error(y_test, y_pred)))

predictions_lin = lin_reg.predict(testData)
lgb = LGBMRegressor(n_estimators = 200, random_state = 22, learning_rate = 0.01)

lgb.fit(X_poly, y_train)

y_pred = lgb.predict(poly_reg.fit_transform(X_test))



#y_pred = np.expm1(y_pred)

#y_test = np.expm1(y_test)



print('LGB RMSE score is', math.sqrt(mean_squared_error(y_test, y_pred)))

predictions = lgb.predict(testData)

#predictions = lgb.predict(poly_reg.fit_transform(test_data))



#predictions = np.expm1(predictions)



fig, ax = plt.subplots(figsize = (10, 10))

lgb_importance(lgb, ax = ax, height = 0.5)

plt.show()
xgb = XGBRegressor(n_estimators = 100, random_state = 22)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)



#y_pred = np.expm1(y_pred)

#y_test = np.expm1(y_test)



print('XGB RMSE score is', math.sqrt(mean_squared_error(y_test, y_pred)))



predictions_xgb = xgb.predict(test_data)



#predictions_xgb = np.expm1(predictions_xgb)
fig, ax = plt.subplots(figsize = (10, 10))

plot_importance(xgb, ax = ax, height = 0.5)

plt.show()
gbr = GradientBoostingRegressor(n_estimators = 200, random_state = 22)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)



#y_pred = np.expm1(y_pred)

#y_test = np.expm1(y_test)



print('GBR RMSE score is', math.sqrt(mean_squared_error(y_test, y_pred)))



predictions_gbr = gbr.predict(test_data)



#predictions_gbr = np.expm1(predictions_gbr)
submission.head()
sub_df1 = pd.DataFrame({'ID': testIDs, 'Upvotes': np.round(predictions)})

sub_df2 = pd.DataFrame({'ID': testIDs, 'Upvotes': np.round(predictions_xgb)})

sub_df3 = pd.DataFrame({'ID': testIDs, 'Upvotes': np.round(predictions_gbr)})

sub_df4 = pd.DataFrame({'ID': testIDs, 'Upvotes': np.round(predictions_lin)})

sub_df1.to_csv('upvotes_lgb_v4.csv', index = False)

sub_df2.to_csv('upvotes_xgb_v4.csv', index = False)

sub_df3.to_csv('upvotes_gbr_v5.csv', index = False)

sub_df4.to_csv('upvotes_poly_v1.csv', index = False)