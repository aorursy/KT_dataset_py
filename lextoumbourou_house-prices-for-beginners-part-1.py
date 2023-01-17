import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from IPython.display import FileLink
df = pd.read_csv('../input/train.csv')
df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
plt.scatter(x=df.GrLivArea, y=df.SalePrice)
plt.title('Sale Price vs Greater Living Area')
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea'] + df['EnclosedPorch'] + df['ScreenPorch']
df.head()
plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.title('Sale Price vs Total SF')
model = LinearRegression()
model.fit(X=df[['TotalSF']], y=df.SalePrice)
model.coef_
model.intercept_
predicted_values = model.intercept_ + df.TotalSF * model.coef_
plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.plot(df.TotalSF, predicted_values, color='red')
plt.title('Sale Price vs Total SF (with predicted vales)')
plt.show()
(-25525.212290) ** 2
math.sqrt(((predicted_values - df.SalePrice) ** 2).mean())
indexes_to_drop = df[(df.TotalSF > 8000) & (df.SalePrice < 400000)].index
df.shape
df.drop(indexes_to_drop, inplace=True)
df.shape
model = LinearRegression()
model.fit(X=df[['TotalSF']], y=df.SalePrice)
preds = model.predict(df[['TotalSF']])
plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.plot(df.TotalSF, preds, color='red')
plt.title('Sale Price vs Total SF (with predicted vales - outliers removed)')
plt.show()
math.sqrt(((preds - df.SalePrice) ** 2).mean())
df['SalePriceLog'] = np.log(df.SalePrice)
model = LinearRegression()
model.fit(X=df[['TotalSF']], y=df.SalePriceLog)
preds = model.predict(df[['TotalSF']])
math.sqrt(((preds - df.SalePriceLog) ** 2).mean())
df.shape
total_sqft_train, total_sqft_val, sale_price_train, sale_price_val = train_test_split(df[['TotalSF']], df.SalePriceLog, test_size=0.2, random_state=42)
total_sqft_train.shape
model = LinearRegression()
model.fit(X=total_sqft_train, y=sale_price_train)
preds = model.predict(total_sqft_val)
math.sqrt(((preds - sale_price_val) ** 2).mean())
test_df = pd.read_csv('../input/test.csv')
test_df.head()
test_df['TotalSF'] = (
    test_df['GrLivArea'] +
    test_df['TotalBsmtSF'].fillna(0) +
    test_df['GarageArea'].fillna(0) +
    test_df['EnclosedPorch'].fillna(0) +
    test_df['ScreenPorch'].fillna(0))
test_preds = model.predict(test_df[['TotalSF']])
submission_df = pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}
)
submission_df.head()
submission_df.to_csv('my_sub.csv', index=False)
FileLink('my_sub.csv')
