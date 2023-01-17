import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
url = '../input/house-prices-advanced-regression-techniques/housetrain.csv'

df = pd.read_csv(url)
df_float = df.select_dtypes(include=['float64']).copy()

df_float.info()
df_float.describe()
df_float['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=False)
df_float['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=False)
df_float['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(), inplace=False)
df_float.info()
lotFrontage = df_float[['LotFrontage']]

salePrice = df['SalePrice']
lr1 = LinearRegression()

lr1.fit(lotFrontage, salePrice)
print(lr1.coef_)

print(lr1.intercept_)

print(mean_squared_error(salePrice, lr1.predict(lotFrontage)))
masVnrAre = df_float[['MasVnrArea']]

salePrice = df['SalePrice']
lr2 = LinearRegression()

lr2.fit(masVnrAre, salePrice)
print(lr2.coef_)

print(lr2.intercept_)

print(mean_squared_error(salePrice, lr2.predict(masVnrAre)))
garageYrBlt = df_float[['GarageYrBlt']]

salePrice = df['SalePrice']
lr3 = LinearRegression()

lr3.fit(garageYrBlt, salePrice)
print(lr3.coef_)

print(lr3.intercept_)

print(mean_squared_error(salePrice, lr3.predict(garageYrBlt)))
f = plt.figure()

f, ax = plt.subplots(1, 3, figsize=(30, 8))



ax = plt.subplot(1, 3, 1)

plt.ylabel('SalePrice')

plt.xlabel('LotFrontage')

ax = plt.scatter(lotFrontage, salePrice)

ax = plt.plot(lotFrontage, lr1.predict(lotFrontage), linewidth=5.0, color='orange')



ax = plt.subplot(1, 3, 2)

plt.ylabel('SalePrice')

plt.xlabel('MasVnrArea')

ax = plt.scatter(masVnrAre, salePrice)

ax = plt.plot(masVnrAre, lr2.predict(masVnrAre), linewidth=5.0, color='orange')



ax = plt.subplot(1, 3, 3)

plt.ylabel('SalePrice')

plt.xlabel('GarageYrBlt')

ax = plt.scatter(garageYrBlt, salePrice)

ax = plt.plot(garageYrBlt, lr3.predict(garageYrBlt), linewidth=5.0, color='orange')



ax = plt.show()
names = ['LotFrontage', 'MasVnrArea',	'GarageYrBlt']

heights = [

           mean_squared_error(salePrice, lr1.predict(lotFrontage)),

           mean_squared_error(salePrice, lr2.predict(masVnrAre)),

           mean_squared_error(salePrice, lr3.predict(garageYrBlt))]



f = plt.figure(figsize=(8,8))

ax = plt.bar(names, heights)