import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
df_train_orig = pd.read_csv('../input/train.csv')
df_train = df_train_orig.copy()

df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF'] + df_train['GarageArea'] + df_train['EnclosedPorch'] + df_train['ScreenPorch']
df_train['SalePrice'] = np.log(df_train['SalePrice'])

df_train['ExterQual'] = df_train.ExterQual.astype('category')
df_train['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['ExterQual'] = df_train['ExterQual'].cat.codes

df_train['Neighborhood'] = df_train['Neighborhood'].astype('category')
dummies = pd.get_dummies(df_train['Neighborhood'])
train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual']], dummies], axis=1)
for i in range(0, 5):
    train_df, train_val, sale_price_train, sale_price_val = train_test_split(
        train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=i)

    model = LinearRegression()
    model.fit(train_df, sale_price_train)

    preds = model.predict(train_val)
    print(f'Val accuracy for iter {i}: {math.sqrt(((preds - sale_price_val)**2).mean())}')
model = LinearRegression()

scores = cross_val_score(model, train_df_concat,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5)
scores = np.sqrt(-scores)
scores
scores.mean()
plt.hist(df_train_orig['LotArea'][df_train_orig['LotArea'] > 0], bins=30)
plt.show()
df_train_orig['LotAreaLog'] = np.log1p(df_train_orig['LotArea'])
plt.hist(df_train_orig['LotAreaLog'][df_train_orig['LotAreaLog'] > 0], bins=30)
plt.show()
train_df_with_lot_val = pd.concat([train_df_concat, df_train_orig[['LotArea']]], axis=1)
train_df_with_log_lot_val = pd.concat([train_df_concat, df_train_orig[['LotAreaLog']]], axis=1)

model = LinearRegression()
scores_lot_val = np.sqrt(
    -cross_val_score(model, train_df_with_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))
scores_log_lot_val = np.sqrt(
    -cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

print(f'Scores with LotArea {scores_lot_val.mean()}')
print(f'Scores with log(LotArea) {scores_log_lot_val.mean()}')
scores_log_lot_val
model = Ridge(alpha=0.01, max_iter=20000)
scores = np.sqrt(-cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))
scores.mean()
model = ElasticNet(max_iter=20000)

grid = GridSearchCV(model, {
    'alpha': [1, 0.1, 0.01, 0.04, 0.001, 0.0001],
    'l1_ratio': [0.0001, 0.001, 0.01, 0.5]
}, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)

grid.fit(train_df_with_log_lot_val, df_train['SalePrice'])
grid.best_params_
math.sqrt(-grid.best_score_)
model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=20000)
scores = np.sqrt(-cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

scores.mean()
model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=20000)
model.fit(train_df_with_log_lot_val, df_train['SalePrice'])
test_df = pd.read_csv('../input/test.csv')

test_df['TotalSF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF'].fillna(0) + test_df['GarageArea'].fillna(0) + test_df['EnclosedPorch'].fillna(0) + test_df['ScreenPorch'].fillna(0)

test_df['ExterQual'] = test_df.ExterQual.astype('category')
test_df['ExterQual'].cat.set_categories(
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True
)
test_df['ExterQual'] = test_df['ExterQual'].cat.codes

test_dummies = pd.get_dummies(test_df['Neighborhood'])
test_df_concat = pd.concat([test_df[['TotalSF', 'OverallQual', 'ExterQual']], test_dummies], axis=1)

test_df['LotAreaLog'] = np.log1p(test_df['LotArea'])
test_df_concat = pd.concat([test_df_concat, test_df[['LotAreaLog']]], axis=1)
test_preds = model.predict(test_df_concat)
pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}).to_csv('elasticnet.csv', index=False)
