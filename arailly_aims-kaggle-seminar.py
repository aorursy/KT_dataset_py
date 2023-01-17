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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# load train data

train = pd.read_csv('../input/train.csv')

print(train.shape)

train.head()
# load test data

test = pd.read_csv('../input/test.csv')

print(test.shape)

test.head()
# load sample submission data

sample_sub = pd.read_csv('../input/sample_submission.csv')

print(sample_sub.shape)

sample_sub.head()
# concatenate train data and test data for preprocess

df = pd.concat([train, test], sort=False).reset_index(drop=True)

print(df.shape)

df.head()
df.tail()
# show all columns

features = df.columns[1:-1]

print(len(features))

features
num_features = train.select_dtypes(include='number').columns[1:-1]

cat_features = train.select_dtypes(exclude='number').columns
import pandas_profiling
pandas_profiling.ProfileReport(df)
target = train['SalePrice']

target.head(10)
target.describe()
%matplotlib inline

plt.figure(figsize=[20, 10])

target.hist(bins=100)
corr_mat = train.loc[:, num_features].corr()

plt.figure(figsize=[15, 15])

sns.heatmap(corr_mat, square=True)
fig = plt.figure(figsize=[30, 30])

plt.tight_layout()



for i, feature in enumerate(num_features):

    ax = fig.add_subplot(6, 6, i+1)

    sns.regplot(x=train.loc[:, feature],

                y=train.loc[:, 'SalePrice'])
fig = plt.figure(figsize=[30, 40])

plt.tight_layout()



for i, feature in enumerate(cat_features):

    ax = fig.add_subplot(9, 5, i+1)

    sns.violinplot(x=df.loc[:, feature],

                   y=df.loc[:, 'SalePrice'])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_features:

    df[col] = df[col].fillna('NULL')

    df[col+'_le'] = le.fit_transform(df[col])
df = df.drop(cat_features, axis=1)
df.head()
le_features = []

for feat in cat_features:

    le_features.append(feat+'_le')
len(le_features)
for feat in num_features:

    df[feat] = df[feat].fillna(-1)
train = df[df['Id'].isin(train['Id'])]

test = df[df['Id'].isin(test['Id'])]
X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = train['SalePrice']



X_test = test.drop(['Id', 'SalePrice'], axis=1)
from sklearn.model_selection import train_test_split
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.3, random_state=42)
reg.fit(X_train_, y_train_)
from sklearn.metrics import mean_squared_error

def metric(y_true, y_pred):

    return mean_squared_error(np.log(y_true), np.log(y_pred)) ** 0.5
pred_train = reg.predict(X_train_)

rmse_train = mean_squared_error(np.log(y_train_), np.log(pred_train))**0.5

rmse_train
pred_train[:5]
y_train_.head()
pred_val = reg.predict(X_val)

rmse_val = mean_squared_error(np.log(y_val), np.log(pred_val))**0.5

rmse_val
pred_test = reg.predict(X_test)

print(pred_test.shape)

pred_test[:5]
sub = pd.read_csv('../input/sample_submission.csv')

print(sub.shape)

sub.head()
sub['SalePrice'] = pred_test

sub.head()
sub.to_csv('submission_ridge_regression.csv', index=False)
from sklearn.model_selection import KFold
def cv(reg, X_train, y_train, X_test):

    kf = KFold(n_splits=5, random_state=42)

    pred_test_mean = np.zeros(sub['SalePrice'].shape)

    for train_index, val_index in kf.split(X_train):

        X_train_train = X_train.iloc[train_index]

        y_train_train = y_train.iloc[train_index]



        X_train_val = X_train.iloc[val_index]

        y_train_val = y_train.iloc[val_index]



        # training on train data

        reg.fit(X_train_train, y_train_train)

        pred_train = reg.predict(X_train_train)

        metric_train = metric(y_train_train, pred_train)

        print('train metric: ', metric_train)



        # evaluate on validation data

        pred_val = reg.predict(X_train_val)

        metric_val = metric(y_train_val, pred_val)

        print('val metric:   ', metric_val)

        print()



        # predict for test data

        pred_test = reg.predict(X_test)

        pred_test_mean += pred_test / kf.get_n_splits()

        

    return pred_test_mean
reg = Ridge(alpha=0.3, random_state=42)

pred_test_mean = cv(reg, X_train, y_train, X_test)
sub['SalePrice'] = pred_test_mean

sub.head()
sub.to_csv('submission_ridge_regression_5f_CV.csv', index=False)
# log tranform

y_train_log = np.log(y_train)

plt.figure(figsize=[20, 10])

plt.hist(y_train_log, bins=50);
reg = Ridge(alpha=0.3, random_state=42)

pred_test_mean = cv(reg, X_train, y_train_log, X_test)
sub['SalePrice'] = np.exp(pred_test_mean)

sub.to_csv('submission_ridge_regression_cv_target_log.csv', index=False)

sub.head()
from sklearn.preprocessing import StandardScaler
# standard scaling

# mean = 0, standard deviation = 1

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))

X_train_scaled.head()
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test))

X_test_scaled.head()
reg = Ridge(alpha=0.3, random_state=42)

pred_test = cv(reg, X_train_scaled, y_train_log, X_test_scaled)
sub['SalePrice'] = np.exp(pred_test)

sub.to_csv('submission_ridge_regression_cv_target_log_scaled_feature.csv', index=False)

sub.head()
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=1000, random_state=42)

pred_test = cv(reg, X_train, y_train_log, X_test)
sub['SalePrice'] = np.exp(pred_test)

sub.to_csv('submission_random_forest_cv_target_log.csv', index=False)

sub.head()
reg.fit(X_train, y_train_log)
feature_importances = reg.feature_importances_

feature_importances
feature_importances = pd.DataFrame([X_train.columns, feature_importances]).T

feature_importances = feature_importances.sort_values(by=1, ascending=False)
plt.figure(figsize=[20, 20])

sns.barplot(x=feature_importances.iloc[:, 1],

            y=feature_importances.iloc[:, 0], orient='h')

plt.tight_layout()

plt.show()