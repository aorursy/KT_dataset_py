
%load_ext autoreload
%autoreload 2

%matplotlib inline

from math import sqrt
import os
from pathlib import Path

from IPython.display import display, FileLink

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
PATH = Path("../input/")
list(PATH.iterdir())
df_raw = pd.read_csv(PATH / 'train.csv')
df_raw.head().transpose()
plt.scatter(x=df_raw['GrLivArea'], y=df_raw['SalePrice'])
plt.title('SALE PRICE vs GR LIV AREA')
plt.show()
idx_to_drop = df_raw[(df_raw['GrLivArea']>4000)].index
df_raw.drop(idx_to_drop, inplace=True)
plt.scatter(x=df_raw['GrLivArea'], y=df_raw['SalePrice'])
plt.title('SALE PRICE vs GR LIV AREA (without outliers)')
plt.show()
df_raw['TotalSF'] = (
    df_raw['BsmtFinSF1'].fillna(0) +
    df_raw['BsmtFinSF2'].fillna(0) +
    df_raw['1stFlrSF'].fillna(0) +
    df_raw['2ndFlrSF'].fillna(0)
)
df_raw.TotalSF.head()
sale_price = df_raw.pop('SalePrice')
sale_price_log = np.log(sale_price)

# We also don't need this.
house_ids = df_raw.pop('Id')
continuous_columns = [
    'BsmtUnfSF',
    'FullBath',
    'LotFrontage',
    'BsmtFullBath',
    '3SsnPorch',
    'BedroomAbvGr',
    'LowQualFinSF',
    'BsmtFinSF1',
    'WoodDeckSF',
    'GarageArea',
    'MiscVal',
    'BsmtHalfBath',
    'HalfBath',
    'EnclosedPorch',
    'ScreenPorch',
    'TotRmsAbvGrd',
    'Fireplaces',
    'KitchenAbvGr',
    'GarageCars',
    '1stFlrSF',
    'BsmtFinSF2',
    'PoolArea',
    '2ndFlrSF',
    'TotalBsmtSF',
    'TotalSF',
    'GrLivArea',
    'LotArea',
    'OpenPorchSF',
    'MasVnrArea'
]

categorical_columns = [col for col in df_raw.columns if col not in continuous_columns]
categorical_columns
assert len(df_raw.columns) == len(categorical_columns + continuous_columns)
for col_name, col in df_raw[categorical_columns].items():
    df_raw[col_name] = col.astype('category').cat.as_ordered()
ordinal_column_data = [
    ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtExposure', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtFinType1', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('BsmtFinType2', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageFinish', ['Unf', 'Rfn', 'Fin']),
    ('GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('PoolQC', ['Fa', 'TA', 'Gd', 'Ex']),
    ('OverallQual', list(range(1, 11))),
    ('OverallCond', list(range(1, 11))),
    ('LandSlope', ['Sev', 'Mod', 'Gtl']),  # Assume less slope is better
    ('Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']),
    ('YearBuilt', list(range(1800, 2018))),
    ('YrSold', list(range(2006, 2018))),
    ('GarageYrBlt', list(range(1900, 2018))),
    ('YearRemodAdd', list(range(1900, 2018)))
]

ordinal_columns = [o[0] for o in ordinal_column_data]

for col, categories in ordinal_column_data:
    df_raw[col].cat.set_categories(categories, ordered=True, inplace=True)
other_cat_columns = [col for col in categorical_columns if col not in ordinal_columns]
assert len(categorical_columns) == len(ordinal_columns + other_cat_columns)
NAs = {}
for col in (
    'GarageArea', 'GarageCars', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea'
):
    NAs[col] = 0
    df_raw[col] = df_raw[col].fillna(0)
    df_raw[f'{col}_na'] = pd.isna(df_raw[col])
for col in continuous_columns:
    if not len(df_raw[df_raw[col].isna()]):
        continue
        
    median = df_raw[col].median()
        
    df_raw[f'{col}_na'] = pd.isna(df_raw[col])
    df_raw[col] = df_raw[col].fillna(median)
    
    NAs[col] = median
skew_feats = df_raw[continuous_columns].apply(skew).sort_values(ascending=False)
skew_feats.head(10)
sns.distplot(df_raw[df_raw['MiscVal'] != 0]['MiscVal'])
skew_feats = skew_feats[abs(skew_feats) > 0.75]

for feat in skew_feats.index:
    df_raw[feat] = np.log1p(df_raw[feat])
sns.distplot(df_raw[df_raw['MiscVal'] != 0]['MiscVal'])
df_numeric = df_raw.copy()
dummies = pd.get_dummies(df_numeric[other_cat_columns], dummy_na=True)
for col_name in categorical_columns:
    # Use +1 to push the -1 NaN value to 0
    df_numeric[col_name] = df_numeric[col_name].cat.codes + 1
df_numeric.drop(other_cat_columns, axis=1, inplace=True)
df_numeric = pd.concat([df_numeric, dummies], axis=1)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = Lasso(alpha=0.0004)
scores = np.sqrt(
    -cross_val_score(model, df_numeric, sale_price_log, cv=kf, scoring='neg_mean_squared_error'))
scores.mean()
final_model  = Lasso(alpha=0.0004)
final_model.fit(df_numeric, sale_price_log)
df_test_raw = pd.read_csv(PATH / 'test.csv')
house_ids = df_test_raw.pop('Id')
df_test_raw['TotalSF'] = (
    df_test_raw['BsmtFinSF1'].fillna(0) +
    df_test_raw['BsmtFinSF2'].fillna(0) +
    df_test_raw['1stFlrSF'].fillna(0) +
    df_test_raw['2ndFlrSF'].fillna(0)
)
for col_name in categorical_columns:
    df_test_raw[col_name] = (
        pd.Categorical(
            df_test_raw[col_name],
            categories=df_raw[col_name].cat.categories,
            ordered=True))
for col in continuous_columns:
    if col not in NAs:
        continue

    df_test_raw[f'{col}_na'] = pd.isna(df_test_raw[col])
    df_test_raw[col] = df_test_raw[col].fillna(NAs[col])
# Handle any other NAs
df_test_raw[continuous_columns] = df_test_raw[continuous_columns].fillna(
    df_test_raw[continuous_columns].median()
)
for feat in skew_feats.index:
    df_test_raw[feat] = np.log1p(df_test_raw[feat])
df_test = df_test_raw.copy()
test_dummies = pd.get_dummies(df_test[other_cat_columns], dummy_na=True)
for col_name in categorical_columns:
    # Use +1 to push the -1 NaN value to 0
    df_test[col_name] = df_test[col_name].cat.codes + 1
df_test.drop(other_cat_columns, axis=1, inplace=True)
df_test = pd.concat([df_test, test_dummies], axis=1)
test_preds = final_model.predict(df_test)
pd.DataFrame(
    {'Id': house_ids, 'SalePrice': np.exp(test_preds)}
).to_csv('output.csv')
