# imports

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")



# preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold



# models

import sklearn.model_selection

from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV

from sklearn.svm import SVR, LinearSVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import cross_val_predict as cvp

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import xgboost as xgb

import lightgbm as lgb



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# data import

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train.tail()
test.head(2)
train.shape
test.shape
train.info()
plt.figure(figsize = (18, 10))

sns.heatmap(train.isnull(), cbar=False)
del_threshold = 0.7

cols_to_del = []

for i in train.columns:

    inulls = train[i].isnull().sum()

    itotal = train[i].isnull().count()

    if inulls/itotal>del_threshold:

        print(i, inulls/itotal)

        cols_to_del.append(i)

print(cols_to_del)
train.drop(columns=cols_to_del, inplace=True)

test.drop(columns=cols_to_del, inplace=True)
train.describe(include='all')
train.columns
# Here's a brief version of what you'll find in the data description file.



# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

# MSSubClass: The building class

# MSZoning: The general zoning classification

# LotFrontage: Linear feet of street connected to property

# LotArea: Lot size in square feet

# Street: Type of road access

# Alley: Type of alley access

# LotShape: General shape of property

# LandContour: Flatness of the property

# Utilities: Type of utilities available

# LotConfig: Lot configuration

# LandSlope: Slope of property

# Neighborhood: Physical locations within Ames city limits

# Condition1: Proximity to main road or railroad

# Condition2: Proximity to main road or railroad (if a second is present)

# BldgType: Type of dwelling

# HouseStyle: Style of dwelling

# OverallQual: Overall material and finish quality

# OverallCond: Overall condition rating

# YearBuilt: Original construction date

# YearRemodAdd: Remodel date

# RoofStyle: Type of roof

# RoofMatl: Roof material

# Exterior1st: Exterior covering on house

# Exterior2nd: Exterior covering on house (if more than one material)

# MasVnrType: Masonry veneer type

# MasVnrArea: Masonry veneer area in square feet

# ExterQual: Exterior material quality

# ExterCond: Present condition of the material on the exterior

# Foundation: Type of foundation

# BsmtQual: Height of the basement

# BsmtCond: General condition of the basement

# BsmtExposure: Walkout or garden level basement walls

# BsmtFinType1: Quality of basement finished area

# BsmtFinSF1: Type 1 finished square feet

# BsmtFinType2: Quality of second finished area (if present)

# BsmtFinSF2: Type 2 finished square feet

# BsmtUnfSF: Unfinished square feet of basement area

# TotalBsmtSF: Total square feet of basement area

# Heating: Type of heating

# HeatingQC: Heating quality and condition

# CentralAir: Central air conditioning

# Electrical: Electrical system

# 1stFlrSF: First Floor square feet

# 2ndFlrSF: Second floor square feet

# LowQualFinSF: Low quality finished square feet (all floors)

# GrLivArea: Above grade (ground) living area square feet

# BsmtFullBath: Basement full bathrooms

# BsmtHalfBath: Basement half bathrooms

# FullBath: Full bathrooms above grade

# HalfBath: Half baths above grade

# Bedroom: Number of bedrooms above basement level

# Kitchen: Number of kitchens

# KitchenQual: Kitchen quality

# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

# Functional: Home functionality rating

# Fireplaces: Number of fireplaces

# FireplaceQu: Fireplace quality

# GarageType: Garage location

# GarageYrBlt: Year garage was built

# GarageFinish: Interior finish of the garage

# GarageCars: Size of garage in car capacity

# GarageArea: Size of garage in square feet

# GarageQual: Garage quality

# GarageCond: Garage condition

# PavedDrive: Paved driveway

# WoodDeckSF: Wood deck area in square feet

# OpenPorchSF: Open porch area in square feet

# EnclosedPorch: Enclosed porch area in square feet

# 3SsnPorch: Three season porch area in square feet

# ScreenPorch: Screen porch area in square feet

# PoolArea: Pool area in square feet

# PoolQC: Pool quality

# Fence: Fence quality

# MiscFeature: Miscellaneous feature not covered in other categories

# MiscVal: $Value of miscellaneous feature

# MoSold: Month Sold

# YrSold: Year Sold

# SaleType: Type of sale

# SaleCondition: Condition of sale
# full description

file = '../input/house-prices-advanced-regression-techniques/data_description.txt'

with open(file, 'r') as f:

    print(f.read())
train.SalePrice.describe()
fig, ax = plt.subplots(figsize=(18, 8))

sns.distplot(train['SalePrice'], label="Price", kde=True, bins=150)

plt.xticks(rotation=45)
_ = sns.boxplot(train.SalePrice)

plt.xticks(rotation=45)
fig, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(train.corr(), annot=False, fmt=".2%", linewidths=.1, cmap="coolwarm", square=True)
# посмотрим поточнее на k самых коррелирующих с ценой признаков

k = 10 

cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

sns.set(font_scale=1.25)

fig, ax = plt.subplots(figsize=(8, 8))

hm = sns.heatmap(train[cols].corr(),

                 cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},

                 cmap="coolwarm")

plt.show()
sns.pairplot(train, 

             vars = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice'] ,

             diag_kind = 'kde')
fig, ax = plt.subplots(figsize=(8, 6))

_ = sns.boxplot(x='OverallQual', y="SalePrice", data=train)
fig, ax = plt.subplots(figsize=(8, 6))

_ = sns.boxplot(x='GarageCars', y="SalePrice", data=train)
fig, ax = plt.subplots(figsize=(8, 6))

_ = sns.boxplot(x='FullBath', y="SalePrice", data=train)
fig, ax = plt.subplots(figsize=(18, 6))

train.YearBuilt.hist(bins=train.YearBuilt.nunique())
fig, ax = plt.subplots(figsize=(20, 8))

_ = sns.boxplot(x='YearBuilt', y="SalePrice", data=train)

plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(20, 8))

_ = sns.boxplot(x='YearRemodAdd', y="SalePrice", data=train)

plt.xticks(rotation=90)
# # среднее и стандартное отклонение

# mean = train.mean(axis=0)

# std = train.std(axis=0)

# # 0 мат ожидание и 1 дисперсию

# train1 = (train - mean)/std

# test1 = (test - mean)/std
fig, ax = plt.subplots(figsize=(10, 6))

sns.distplot(np.log(train['SalePrice']), label="Price", kde=True, bins=150)
train1 = train.copy()

test1 = test.copy()

train1['SalePrice'] = np.log(train1['SalePrice'])
train1['GrLivArea'] = np.log(train1['GrLivArea'])

train1['TotalBsmtSF'] = np.log(train1['TotalBsmtSF'])

test1['GrLivArea'] = np.log(test1['GrLivArea'])

test1['TotalBsmtSF'] = np.log(test1['TotalBsmtSF'])
fig, ax = plt.subplots(figsize=(10, 6))

sns.distplot(train1['GrLivArea'], label="GrLivArea", kde=True, bins=150)
fig, ax = plt.subplots(figsize=(10, 6))

sns.distplot(train1[train1['TotalBsmtSF']>0]['TotalBsmtSF'], label="TotalBsmtSF", kde=True, bins=150)
# train1 = pd.get_dummies(train1)

# train1.head(2)

# test1 = pd.get_dummies(test1)
from sklearn.preprocessing import LabelEncoder

# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

train2 = train1.copy()

test2  = test1.copy()

features = train2.columns.values.tolist()

for col in features:

    if train2[col].dtype in numerics: continue

    categorical_columns.append(col)

print('cat cols:', categorical_columns)

# Encoding categorical features

for col in categorical_columns:

    if col in train2.columns:

        le = LabelEncoder()

        le.fit(list(train2[col].astype(str).values))

        train2[col] = le.transform(list(train2[col].astype(str).values))



categorical_columns = []

features = test2.columns.values.tolist()

for col in features:

    if test2[col].dtype in numerics: continue

    categorical_columns.append(col)

# Encoding categorical features

for col in categorical_columns:

    if col in test2.columns:

        le = LabelEncoder()

        le.fit(list(test[col].astype(str).values))

        test2[col] = le.transform(list(test2[col].astype(str).values))
train2.replace([np.inf, -np.inf], np.nan, inplace=True)

train2.fillna(0, inplace=True)

test2.replace([np.inf, -np.inf], np.nan, inplace=True)

test2.fillna(0, inplace=True)
X = train2.drop(columns=['SalePrice'])

y = np.asarray(train2['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
from sklearn.utils.multiclass import type_of_target

type_of_target(y)
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

type_of_target(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# Создаем списки для сохранения точности на тренировочном и тестовом датасете

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

trees_grid = [5, 10, 15, 20, 30, 50, 75, 100, 200, 400]



# Обучаем на тренировочном датасете

for ntrees in trees_grid:

    rfc = RandomForestRegressor(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)



train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100, 

                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
plt.style.use('ggplot')

%matplotlib inline



fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.80,1.0])

ax.set_ylabel("Accuracy")

ax.set_xlabel("N_estimators")
poly = PolynomialFeatures(2)

X_poly = poly.fit_transform(X)

X_poly_df = pd.DataFrame(X_poly, columns = poly.get_feature_names(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=146)
X_poly_df.head(2)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# Создаем списки для сохранения точности на тренировочном и тестовом датасете

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]



# Обучаем на тренировочном датасете

for ntrees in trees_grid:

    rfc = RandomForestRegressor(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(X_poly_df, y):

        X_train, X_test = X_poly_df.iloc[train_index], X_poly_df.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)



train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100, 

                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
plt.style.use('ggplot')

%matplotlib inline



fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.80,1.0])

ax.set_ylabel("Accuracy")

ax.set_xlabel("N_estimators")
test
rfc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)

rfc.fit(X, y)

predict = rfc.predict(test2)
predict
submission = pd.DataFrame({"Id": test.index,

                           "SalePrice": predict

                          })

submission.to_csv('submission.csv', index=False)
# xgb = XGBRegressor( booster='gbtree', colsample_bylevel=1,

#                    colsample_bynode=1, colsample_bytree=0.6, gamma=0,

#                    importance_type='gain', learning_rate=0.01, max_delta_step=0,

#                    max_depth=4, min_child_weight=1.5, n_estimators=2400,

#                    n_jobs=1, nthread=None, objective='reg:linear',

#                    reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

#                    silent=None, subsample=0.8, verbosity=1

#                  )
# xgb.fit(X_train, y_train)
# predict = xgb.predict(X_test)
# import math
# print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))
# xgb.fit(X, y)

# predict0 = xgb.predict(test)
# submission = pd.DataFrame({"Id": test.index,

#                            "SalePrice": predict0

#                           })

# submission.to_csv('submission.csv', index=False)
# # перебираем глубину

# # перебираем мин кол-во для разделения

# # максимально кол-во признаков для более случайной выборки

# param_grid = {'max_depth': [i for i in [15, 55, 105, 155, 200]],

#               'min_samples_split': [i for i in range(2, 4)],

#               #'max_features': [2, len(X.columns)-1]

#              }



# # инициализируем случайный лес с перебором по кросс-вал на выбранных выше праметрах

# gs = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid, verbose=2)

# gs.fit(X_train, y_train)



# # best_params_ содержит в себе лучшие подобранные параметры, best_score_ лучшее качество

# print()

# gs.best_params_, gs.best_score_