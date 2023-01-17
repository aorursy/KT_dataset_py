import math
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
df_train = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

df_train.head()
     # LotFrontage, NA --> median
df_train['LotFrontage'] = df_train.LotFrontage.fillna(df_train['LotFrontage'].median())
test_df['LotFrontage'] = test_df.LotFrontage.fillna(test_df['LotFrontage'].median())
    
    # BsmtFinSF1, NA --> 0
df_train['BsmtFinSF1'] = df_train['BsmtFinSF1'].fillna(0)
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(0)
df_train.isna().sum().sort_values(ascending=False)
test_df.isna().sum().sort_values(ascending=False)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF'] + df_train['GarageArea'] + df_train['EnclosedPorch'] + df_train['ScreenPorch'] + df_train['OpenPorchSF']

test_df['TotalSF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF'].fillna(0) + test_df['GarageArea'].fillna(0) + test_df['EnclosedPorch'].fillna(0) + test_df['ScreenPorch'].fillna(0) + test_df['OpenPorchSF'].fillna(0)
df_train['TotalSF'].head()
train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF', 'OverallQual']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
print(f'Intercept: {model.intercept_}, coefficients: {model.coef_}')
df_train['ExterQual'] = df_train.ExterQual.astype('category')
df_train['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['ExterQual'] = df_train['ExterQual'].cat.codes

test_df['ExterQual'] = test_df.ExterQual.astype('category')
test_df['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
test_df['ExterQual'] = test_df['ExterQual'].cat.codes
df_train['ExterCond'] = df_train.ExterCond.astype('category')
df_train['ExterCond'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['ExterCond'] = df_train['ExterCond'].cat.codes

test_df['ExterCond'] = test_df.ExterCond.astype('category')
test_df['ExterCond'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
test_df['ExterCond'] = test_df['ExterCond'].cat.codes
df_train['HeatingQC'] = df_train.HeatingQC.astype('category')
df_train['HeatingQC'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['HeatingQC'] = df_train['HeatingQC'].cat.codes

test_df['HeatingQC'] = test_df.HeatingQC.astype('category')
test_df['HeatingQC'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
test_df['HeatingQC'] = test_df['HeatingQC'].cat.codes
df_train['PavedDrive'] = df_train.PavedDrive.astype('category')
df_train['PavedDrive'].cat.set_categories(['N', 'P', 'Y'], ordered=True, inplace=True)
df_train['PavedDrive'] = df_train['PavedDrive'].cat.codes

test_df['PavedDrive'] = test_df.PavedDrive.astype('category')
test_df['PavedDrive'].cat.set_categories(['N', 'P', 'Y'], ordered=True, inplace=True)
test_df['PavedDrive'] = test_df['PavedDrive'].cat.codes

# LandSlope 
df_train['LandSlope'] = df_train.LandSlope.astype('category')
df_train['LandSlope'].cat.set_categories(['Sev', 'Mod', 'Gtl'], ordered=True, inplace=True)
df_train['LandSlope'] = df_train['LandSlope'].cat.codes

test_df['LandSlope'] = test_df.LandSlope.astype('category')
test_df['LandSlope'].cat.set_categories(['Sev', 'Mod', 'Gtl'], ordered=True, inplace=True)
test_df['LandSlope'] = test_df['LandSlope'].cat.codes
 # Basement (BsmtExposure)  / BsmtFinSF1- Square / BsmtFullBath - 0,1 
df_train['BsmtExposure'] = df_train.BsmtExposure.astype('category')
df_train['BsmtExposure'].cat.set_categories(['NA','No', 'Mn', 'TA', 'Gd'], ordered=True, inplace=True)
df_train['BsmtExposure'] = df_train['BsmtExposure'].cat.codes

test_df['BsmtExposure'] = test_df.BsmtExposure.astype('category')
test_df['BsmtExposure'].cat.set_categories(['NA','No', 'Mn', 'TA', 'Gd'], ordered=True, inplace=True)
test_df['BsmtExposure'] = test_df['BsmtExposure'].cat.codes

# FireplaceQu
df_train['FireplaceQu'] = df_train.FireplaceQu.astype('category')
df_train['FireplaceQu'].cat.set_categories(['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['FireplaceQu'] = df_train['FireplaceQu'].cat.codes

test_df['FireplaceQu'] = test_df.FireplaceQu.astype('category')
test_df['FireplaceQu'].cat.set_categories(['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
test_df['FireplaceQu'] = test_df['FireplaceQu'].cat.codes
df_train['KitchenQual'] = df_train.KitchenQual.astype('category')
df_train['KitchenQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['KitchenQual'] = df_train['KitchenQual'].cat.codes

test_df['KitchenQual'] = test_df.KitchenQual.astype('category')
test_df['KitchenQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
test_df['KitchenQual'] = test_df['KitchenQual'].cat.codes
# SaleType
df_train['SaleType'] = df_train.SaleType.astype('category')
df_train['SaleType'].cat.set_categories(['Oth', 'ConLD', 'ConLI', 'ConLw', 'Con', 'COD', 'New', 'VWD','CWD','WD'], ordered=True, inplace=True)
df_train['SaleType'] = df_train['SaleType'].cat.codes

test_df['SaleType'] = test_df.SaleType.astype('category')
test_df['SaleType'].cat.set_categories(['Oth', 'ConLD', 'ConLI', 'ConLw', 'Con', 'COD', 'New', 'VWD','CWD','WD'], ordered=True, inplace=True)
test_df['SaleType'] = test_df['SaleType'].cat.codes
# Test
# CentralAir
# Foundation
# SaleCondition
# Fireplace NA-->0, other categorical
# GarageCond NA-->0, other categorical
# GarageQualNA-->0, other categorical
df_train['Neighborhood'] = df_train['Neighborhood'].astype('category')
dummies = pd.get_dummies(df_train['Neighborhood']) 

test_dummies = pd.get_dummies(test_df['Neighborhood']) 

# + test_df['BsmtFullBath'] # BsmtFullBath?
# New model (with categorical variables + dummy variables)
train_df_concat = pd.concat([df_train[['TotalSF','OverallQual', 'ExterQual','ExterCond','HeatingQC','PavedDrive','BsmtExposure','KitchenQual','SaleType', 'BsmtFinSF1','FireplaceQu','OverallCond','YearRemodAdd','LotArea']], dummies], axis=1)

train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    train_df_concat, df_train['SalePrice'], test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual','ExterCond','HeatingQC','PavedDrive','BsmtExposure','KitchenQual','SaleType', 'BsmtFinSF1','FireplaceQu','OverallCond','YearRemodAdd','LotArea']], dummies], axis=1)

model = ElasticNet(alpha=0.00015)
scores = np.sqrt(-cross_val_score(model, train_df_concat,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

scores.mean()
model = ElasticNet(max_iter=20000)

grid = GridSearchCV(model, {
    'alpha': [1, 0.1, 0.01, 0.04, 0.001, 0.0001,0.00015],
    'l1_ratio': [0.0001, 0.001, 0.01, 0.5, 0.00001]
}, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid.fit(train_df_concat, df_train['SalePrice'])
grid.best_params_
math.sqrt(-grid.best_score_)
train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual','ExterCond','HeatingQC','PavedDrive','BsmtExposure','KitchenQual','SaleType', 'BsmtFinSF1','FireplaceQu','OverallCond','YearRemodAdd','LotArea']], dummies], axis=1)
model = ElasticNet(alpha=0.00015, l1_ratio=0.00001, max_iter=20000)
scores = np.sqrt(-cross_val_score(model, train_df_concat,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

scores.mean()
model = ElasticNet(alpha=0.00015, l1_ratio=0.00001, max_iter=20000)
model.fit(train_df_concat, df_train['SalePrice'])
np.log(1 + df_train['MiscVal'][df_train['MiscVal']])

test_df_concat = pd.concat([test_df[['TotalSF', 'OverallQual', 'ExterQual','ExterCond','HeatingQC','PavedDrive','BsmtExposure','KitchenQual','SaleType', 'BsmtFinSF1','FireplaceQu','OverallCond','YearRemodAdd','LotArea']], test_dummies], axis=1)
test_preds = model.predict(test_df_concat)
pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}).to_csv('my_sub_more_features6.csv', index=False)
