import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df_train = pd.read_csv('../input/train.csv')
# Shift + tab to see docs.
df_train.head()
df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF'] + df_train['GarageArea'] + df_train['EnclosedPorch'] + df_train['ScreenPorch']
df_train['TotalSF'].head()
df_train['SalePrice'] = np.log(df_train['SalePrice'])
train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF', 'OverallQual']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
print(f'Intercept: {model.intercept_}, coefficients: {model.coef_}')
df_train['ExterQual'] = df_train.ExterQual.astype('category')

# Set the ordering of the category.
df_train['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)

# The codes is a column with the category string mapped to a number.
df_train['ExterQual'] = df_train['ExterQual'].cat.codes
train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    df_train[['TotalSF', 'OverallQual', 'ExterQual']], df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
df_train['Neighborhood'] = df_train['Neighborhood'].astype('category')
dummies = pd.get_dummies(df_train['Neighborhood'])
train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual']], dummies], axis=1)
train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
df_train.isna().sum().sort_values(ascending=False).head(20)
df_train['LotFrontage_isna'] = df_train.LotFrontage.isna()
df_train['LotFrontage'] = df_train.LotFrontage.fillna(df_train['LotFrontage'].median())
lot_frontage_na = df_train['LotFrontage'].median()
train_df_concat = pd.concat([
    df_train[['TotalSF', 'OverallQual', 'ExterQual', 'LotFrontage', 'LotFrontage_isna']], dummies], axis=1)

train_df, train_val, sale_price_train, sale_price_val = train_test_split(
    train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_df, sale_price_train)

preds = model.predict(train_val)
math.sqrt(((preds - sale_price_val)**2).mean())
test_df = pd.read_csv('../input/test.csv')

test_df['TotalSF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF'].fillna(0) + test_df['GarageArea'].fillna(0) + test_df['EnclosedPorch'].fillna(0) + test_df['ScreenPorch'].fillna(0)

test_df['ExterQual'] = test_df.ExterQual.astype('category')
test_df['ExterQual'].cat.set_categories(
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True
)
test_df['ExterQual'] = test_df['ExterQual'].cat.codes

test_df['LotFrontage_isna'] = test_df['LotFrontage'].isna()
test_df['LotFrontage'] = lot_frontage_na

test_dummies = pd.get_dummies(test_df['Neighborhood'])
test_df_concat = pd.concat([test_df[['TotalSF', 'OverallQual', 'ExterQual', 'LotFrontage', 'LotFrontage_isna']], test_dummies], axis=1)
test_preds = model.predict(test_df_concat)
pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}).to_csv('my_sub_more_features.csv', index=False)
