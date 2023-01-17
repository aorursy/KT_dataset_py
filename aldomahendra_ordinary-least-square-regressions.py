import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

import missingno

import os

sns.set()
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.head()
df.shape
missingno.matrix(df)
df.info()
data = df[['price', 'bedrooms', 'bathrooms','sqft_living', 'sqft_lot', 'floors', 'condition', 'grade','yr_built']]
data = data[:100]
data.head()
data.describe()
y = data['price']

x1 = data[['sqft_living', 'condition', 'grade','yr_built']]
plt.scatter(data['sqft_living'], y)

plt.xlabel('size', fontsize=20)

plt.ylabel('price', fontsize=20)

plt.show()
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
import pandas as pd

kc_house_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")