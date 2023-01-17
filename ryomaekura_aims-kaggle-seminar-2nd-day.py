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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(train.shape)

train.head()
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(test.shape)

test.head()
sample_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print(sample_sub.shape)

sample_sub.head()
df = pd.concat([train, test], sort=False).reset_index(drop=True)

print(df.shape)

df.head()
df.tail()
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
reg = Ridge(alpha=0.1, random_state=42)
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
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

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



        reg.fit(X_train_train, y_train_train)

        pred_train = reg.predict(X_train_train)

        metric_train = metric(y_train_train, pred_train)

        print('train metric: ', metric_train)



        pred_val = reg.predict(X_train_val)

        metric_val = metric(y_train_val, pred_val)

        print('val metric:   ', metric_val)

        print()



        pred_test = reg.predict(X_test)

        pred_test_mean += pred_test / kf.get_n_splits()

        

    return pred_test_mean
reg = Ridge(alpha=0.3, random_state=42)

pred_test_mean = cv(reg, X_train, y_train, X_test)
sub['SalePrice'] = pred_test_mean

sub.head()
sub.to_csv('submission_ridge_regression_5f_CV.csv', index=False)
y_train_log = np.log(y_train)

plt.figure(figsize=[20, 10])

plt.hist(y_train_log, bins=50);
reg = Ridge(alpha=0.3, random_state=42)

pred_test_mean = cv(reg, X_train, y_train_log, X_test)
sub['SalePrice'] = np.exp(pred_test_mean)

sub.to_csv('submission_ridge_regression_cv_target_log.csv', index=False)

sub.head()
from sklearn.preprocessing import StandardScaler
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
import pandas as pd

import numpy as np



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print("Train Shape: " + str(train.shape))

print("Test Shape: " + str(test.shape))
y = train['SalePrice']

train.drop("SalePrice", axis = 1, inplace = True)
from sklearn import preprocessing

#from sklearn.preprocessing import LabelEncoder

import datetime



ntrain = train.shape[0]



all_data = pd.concat((train, test)).reset_index(drop=True)

print("all_data size is : {}".format(all_data.shape))



def ordinal_encode(df, col, order_list):

    df[col] = df[col].astype('category', ordered=True, categories=order_list).cat.codes

    return df



def label_encode(df, col):

    for c in col:

        #print(str(c))

        encoder = preprocessing.LabelEncoder()

        df[c] = encoder.fit_transform(df[c].astype(str))

    return df 



def split_all_data(all_data, ntrain):

    print(('Split all_data back to train and test: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

    train_df = all_data[:ntrain]

    test_df = all_data[ntrain:]

    return train_df, test_df



"""

NOW START ENCODING 1. ORDINALS

"""

print(('Ordinal Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po'] #This applies to a few different columns

cols = ['KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC']

for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)



order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'] #This applies to a few different columns

cols = ['BsmtQual', 'BsmtCond']

for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)



order_list = ['Gd', 'Av', 'Mn', 'No', 'NA']

cols = ['BsmtExposure', 'FireplaceQu', 'GarageQual', 'GarageCond']



for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)

    

order_list = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']

cols = ['Functional']

for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)

    

order_list = ['Fin', 'RFn', 'Unf', 'NA']

cols = ['GarageFinish']

for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)

    

order_list = ['Ex', 'Gd', 'TA', 'Fa', 'NA'] 

cols = ['PoolQC']

for col in cols:

    all_data = ordinal_encode(all_data, col, order_list)



"""

ENCODE 2. NON-ORDINAL LABELS

"""

print(('Label Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

cols_to_label_encode = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 

                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 

                       'Electrical', 'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

all_data = label_encode(all_data, cols_to_label_encode)



train, test = split_all_data(all_data, ntrain)



print("Train Shape: " + str(train.shape))

print("Test Shape: " + str(test.shape))
def score_transformer(y):

    y = np.log(y)

    

    return y



y = score_transformer(y)
all_data = pd.concat((train, test)).reset_index(drop=True)



all_data['fe.sum.GrLivArea_BsmtFinSF1_BsmtFinSF2'] = all_data['GrLivArea'] + all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] 

all_data['fe.sum.OverallQual_Overall_Cond'] = all_data['OverallQual'] + all_data['OverallCond']

all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']

all_data['fe.sum.KitchenQual_ExterQual'] = all_data['KitchenQual'] + all_data['ExterQual']

all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']

all_data['fe.ratio.1stFlrSF_2ndFlrSF'] = all_data['1stFlrSF'] / all_data['2ndFlrSF']

all_data['fe.ratio.BedroomAbvGr_GrLivArea'] = all_data['BedroomAbvGr'] / all_data['GrLivArea']
train_features = list(all_data)

#Id should be removed for modelling

train_features = [e for e in train_features if e not in ('ExterQual', 'Condition2', 'GarageCond', 'Street', 'Alley', 'PoolArea', 'PoolQC', 'Utilities', 

                                                         'GarageQual', 'MiscVal', 'MiscFeature')]



train, test = split_all_data(all_data, ntrain)



train_features.remove('Id')
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error



nfolds=5

kf = KFold(n_splits=nfolds, shuffle=True, random_state=37) #33 originally

y_valid_pred = 0*y

y_valid_pred_cat = 0*y

fold_scores = [0] * nfolds

fold_scores_cat = [0] * nfolds



importances = pd.DataFrame()

oof_reg_preds = np.zeros(train.shape[0])

sub_reg_preds = np.zeros(test.shape[0])

sub_reg_preds_cat = np.zeros(test.shape[0])
for fold_, (train_index, val_index) in enumerate(kf.split(train, y)):

    trn_x, trn_y = train[train_features].iloc[train_index], y.iloc[train_index]

    val_x, val_y = train[train_features].iloc[val_index], y.iloc[val_index]

    

    reg = LGBMRegressor(

        num_leaves=10,

        max_depth=3,

        min_child_weight=50,

        learning_rate=0.1,

        n_estimators=1000,

        #min_split_gain=0.01,

        #gamma=100,

        reg_alpha=0.01,

        reg_lambda=5,

        subsample=1,

        colsample_bytree=0.5,

        random_state=2

    )

    

    reg.fit(

        trn_x, trn_y,

        eval_set=[(val_x, val_y)],

        early_stopping_rounds=20,

        verbose=100,

        eval_metric='rmse'

    )    

    imp_df = pd.DataFrame()

    imp_df['feature'] = train_features

    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')



    imp_df['fold'] = fold_ + 1

    importances = pd.concat([importances, imp_df], axis=0, sort=False)



    y_valid_pred.iloc[val_index] = reg.predict(val_x, num_iteration=reg.best_iteration_)

    y_valid_pred[y_valid_pred < 0] = 0

    fold_score = reg.best_score_['valid_0']['rmse']

    fold_scores[fold_] = fold_score

    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)

    _preds[_preds < 0] = 0

    sub_reg_preds += _preds / nfolds

    

print("LightGBM CV RMSE: " + str(mean_squared_error(y, y_valid_pred) ** .5))

print("LightGBM CV standard deviation: " + str(np.std(fold_scores)))
import seaborn as sns

import warnings

#cat_rgr.fit(X_train, y_train, eval_set=(X_valid, y_valid), logging_level='Verbose', plot=False)

warnings.simplefilter('ignore', FutureWarning)



importances['gain_log'] = np.log1p(importances['gain'])

mean_gain = importances[['gain', 'feature']].groupby('feature').mean()

importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])



plt.figure(figsize=(10, 14))

sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
sub_reg_preds = np.exp(sub_reg_preds)

test.is_copy = False

test.loc[:,'SalePrice'] = sub_reg_preds

test[['Id', 'SalePrice']].to_csv("lightgbm_fold5.csv", index=False)
import pandas as pd



model1 = pd.read_csv("../input/predicted-data/lightgbm_fold5.csv")

model2 = pd.read_csv("../input/predicted-data/submission_random_forest_cv_target_log.csv")

model3 = pd.read_csv("../input/predicted-data/submission_ridge_regression_5f_CV.csv")



blend_model = (model1["SalePrice"] + model2["SalePrice"] + model3["SalePrice"]) /3



sub['SalePrice'] = blend_model



sub.to_csv("blend_model.csv", index = False)