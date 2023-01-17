# helper library

import os

import pandas as pd

import numpy as np

#visualization module

import matplotlib.pyplot as plt

import seaborn as sns

#data preprocessing and evaluation

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error

from sklearn.model_selection import KFold, cross_val_score, train_test_split

#data modeling

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor, XGBRFRegressor
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
!cat /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
train_data.head()
target_column = 'SalePrice'
sns.distplot(train_data['SalePrice'])
corr = train_data.corr()[['SalePrice']].sort_values(by = 'SalePrice', ascending = False)

corr.head(5)
sns.catplot(x = 'OverallQual', y = 'SalePrice', data = train_data, kind = 'box')

plt.title('OverallQual vs SalePrice')
plt.subplot(1, 1, 1)

sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train_data)
train_data = train_data[train_data.GrLivArea < 4000]
plt.subplot(1, 1, 1)

sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train_data)
all_data = pd.concat([train_data, test_data])
null_df = pd.DataFrame(all_data.isnull().sum())

null_df.columns = ['null_count']

null_df['percent'] = null_df['null_count'] / all_data.shape[0]

#null_df = null_df.loc[null_df.null_count > 0]

null_df = null_df.sort_values(by = 'percent', ascending = False)
null_df.head(20)
categorical_column = all_data.columns[all_data.dtypes == 'object']

df = null_df.head(null_df.shape[0]).T

df[categorical_column].T.sort_values(by = 'percent', ascending = False).head(23)
none_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',

       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtCond',

       'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']

for col in none_columns:

    all_data[col] = all_data[col].fillna('None')

sns.countplot(x = 'MSZoning', data = all_data)

sns.catplot(x = 'MSZoning', y = 'SalePrice', data = train_data)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
column = 'Utilities'

sns.countplot(x = column, data = all_data)

sns.catplot(x = column, y = 'SalePrice', data = train_data)
all_data = all_data.drop([column], axis = 1)
column = 'Functional'

sns.countplot(x = column, data = all_data)

sns.catplot(x = column, y = 'SalePrice', data = train_data)
all_data[column] = all_data[column].fillna(all_data[column].mode()[0])
column = 'Exterior2nd'

sns.countplot(x = column, data = all_data)

sns.catplot(x = column, y = 'SalePrice', data = train_data)
for column in ['Exterior2nd', 'Exterior1st', 'SaleType', 'KitchenQual', 'Electrical']:

    all_data[column] = all_data[column].fillna(all_data[column].mode()[0])
null_df.head(len(null_df))
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
column = 'MasVnrArea'

sns.distplot(all_data[column])
sns.scatterplot(x = column, y = 'SalePrice', data = all_data, hue = 'MasVnrType')
all_data[column] = all_data[column].fillna(0)
for col in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtUnfSF', 'TotalBsmtSF']:

    all_data[col] = all_data[col].fillna(0)
for col in ['GarageCars', 'GarageArea']:

    all_data[col] = all_data[col].fillna(0)
column = 'LotFrontage'

sns.scatterplot(x = 'LotFrontage', y = 'SalePrice', data = all_data)
sns.catplot(x = 'LotFrontage', data = all_data, col = 'Neighborhood')
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
all_data.isnull().sum()
tmp = all_data.copy()

for col in tmp.columns[tmp.dtypes == 'object']:

    tmp[col] = LabelEncoder().fit_transform(tmp[col])

corr = tmp.corr()

plt.figure(figsize = (15, 15))

sns.heatmap(corr)
corr[['SalePrice']].sort_values(by = 'SalePrice', ascending = False)
sns.catplot(x = 'OverallQual', y = 'SalePrice', data = all_data, kind = 'box')
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['GrLivArea']
all_data.head()
for col in all_data.columns[all_data.dtypes == 'object']:

    all_data[col] = LabelEncoder().fit_transform(all_data[col])
columns = ['LotFrontage','MSSubClass', 'MSZoning', 'LotArea', 'Street', 'Alley',

       'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',

       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',

       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual',

       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',

       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',

       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',

       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice', 'TotalSF']
train_data = all_data[columns][~all_data.SalePrice.isnull()]

test_data = all_data[columns][all_data.SalePrice.isnull()]
x,y = train_data.drop(['SalePrice'], axis = 1), train_data['SalePrice']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.20, random_state = 1)

x_test = test_data[[col for col in columns if col != target_column]]
evaluation = pd.DataFrame({'model':[],'details':[],'score':[],'mse':[],'mae':[],'rmse':[],'rmsle':[]})
lr = LinearRegression()

lr.fit(x_train, y_train)

score = lr.score(x_val, y_val)

y_pred = lr.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['LinearRegression','-',score,mse,mae,rmse,rmsle]
rf_param = {'n_estimators':100, 'criterion':'mse', 'ccp_alpha':0.10}

rf = RandomForestRegressor(**rf_param)

rf.fit(x_train, y_train)

score = rf.score(x_val, y_val)

y_pred = rf.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['RandomForestRegressor','n_estimators=1000',score,mse,mae,rmse,rmsle]
param = {'n_estimators' : 100, 'booster': 'dart','gamma':0.1, 'learning_rate' : 0.1, 'max_depth': 3, 'objective':'reg:squarederror', 'random_state' : 10, 'reg_lambda' :0.1, 'base_score' : 0.7}

xgb = XGBRegressor(**param)

xgb.fit(x_train, y_train)

print(xgb)

score = xgb.score(x_val, y_val)

y_pred = xgb.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['XGBRegressor','{}'.format(param),score,mse,mae,rmse,rmsle]
alpha = 200.0

ridge = Ridge(alpha = 150)

ridge.fit(x_train, y_train)

score = ridge.score(x_val, y_val)

y_pred = ridge.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['RidgeRegressor','alpha={}'.format(alpha),score,mse,mae,rmse,rmsle]
from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor
catboost = CatBoostRegressor(verbose = False)

catboost.fit(x_train, y_train)

print(catboost)

score = catboost.score(x_val, y_val)

y_pred = catboost.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['CatBoostRegressor','_',score,mse,mae,rmse,rmsle]
lgb = LGBMRegressor()

lgb.fit(x_train, y_train)

score = lgb.score(x_val, y_val)

y_pred = lgb.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['LGBMRegressor','_',score,mse,mae,rmse,rmsle]
class AveraginModels:

    def __init__(self, models):

        self.models = models

    def fit(self, x, y):

        for model in self.models:

            model.fit(x, y)

    def predict(self, x):

        prediction = np.vstack([model.predict(x) for model in self.models])

        prediction = prediction.mean(axis = 0)

        return prediction

    def score(self, x, y):

        score = np.vstack([model.score(x,y) for model in self.models])

        score = score.mean()

        return score
models = [rf, xgb, ridge, catboost, lgb]

avg_model = AveraginModels(models)

avg_model.fit(x_train, y_train)

score = avg_model.score(x_val, y_val)

y_pred = avg_model.predict(x_val)



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('score: {}\nmse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(score,mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['AveragingModel','_',score,mse,mae,rmse,rmsle]
class StackingModels:

    def __init__(self, base_models, meta_model, n_folds):

        self.base_models = base_models

        self.meta_model = meta_model

        self.kfold = KFold(n_folds, random_state=11, shuffle = True)



    def fit(self, x, y):

        hold_out_prediction = np.zeros((x.shape[0], len(self.base_models)))

        for index, model in enumerate(self.base_models):

            for train, holdout in self.kfold.split(x):

                x_train, x_holdout, y_train = x.iloc[train], x.iloc[holdout], y.iloc[train]

                model.fit(x_train, y_train)

                hold_out_prediction[holdout, index] = model.predict(x_holdout)

        self.meta_model.fit(hold_out_prediction, y)



    def predict(self, x):

        prediction = np.zeros((x.shape[0], len(self.base_models)))

        for index, model in enumerate(self.base_models):

            prediction[:, index] = model.predict(x)

        prediction = self.meta_model.predict(prediction)

        return prediction
stack_model = StackingModels([xgb, ridge, catboost, lgb], lr, 5)

stack_model.fit(x_train,y_train)

y_pred = stack_model.predict(x_val)

np.sqrt(mean_squared_log_error(y_pred, y_val))



#metrics

mse  = mean_squared_error(y_pred, y_val)

rmse = np.sqrt(mse)

mae  = mean_absolute_error(y_pred, y_val)

rmsle = np.sqrt(mean_squared_log_error(y_pred, y_val))

print('mse: {}\nmae: {}\nrmse: {}\nrmsle: {}'.format(mse,mae,rmse,rmsle))



evaluation.loc[evaluation.shape[0]] = ['StackingModel','_',_,mse,mae,rmse,rmsle]
evaluation.head(evaluation.shape[0])
sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



y_pred = stack_model.predict(x_test)

sample.SalePrice = y_pred

sample.to_csv('submission.csv', index=False)