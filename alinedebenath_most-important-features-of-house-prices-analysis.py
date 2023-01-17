import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.formula.api import ols

import numpy as np
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data.head()
data.shape
data.columns
data.dtypes
data.isnull().sum().sort_values(ascending=False)[:20]
data = data.drop(columns=['Id','Street'],axis=1)
data = data.drop(columns=['PoolQC','MiscFeature'],axis=1)
data['TotalBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
for col in ('FireplaceQu',

            'Fence',

            'Alley',

            'GarageType', 

            'GarageFinish', 

            'GarageQual', 

            'GarageCond'):

    data[col]=data[col].fillna('None')
data.loc[data['GarageType']!='None', "GarageYrBlt"] = data["GarageYrBlt"].fillna(data['GarageYrBlt'].median())

data["GarageYrBlt"]=data["GarageYrBlt"].fillna(0)
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
data['YrSold'] = data['YrSold'].astype(str)

data['OverallCond'] = data['OverallCond'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)

data['MSSubClass'] = data['MSSubClass'].apply(str)
encoded_data = pd.get_dummies(data)

encoded_data.head()
data['SalePrice'].describe()
data['SalePrice'].hist(bins=50)
encoded_data['SalePrice_skewed'] = np.log1p(data['SalePrice']) 

encoded_data['SalePrice_skewed'].hist(bins=50)
encoded_data[encoded_data.columns[1:]].corr()['SalePrice_skewed'][:].sort_values(ascending=False)[2:12]
plt.figure(figsize=(20,8))

sns.regplot(x='GrLivArea', y="SalePrice_skewed", data=encoded_data, color='green')
encoded_data[(encoded_data['GrLivArea']> 4000) & (encoded_data['SalePrice_skewed']<13)]
encoded_data.drop([523,1298], inplace=True)
plt.figure(figsize=(10,8))

sns.boxplot(x="OverallQual", y="SalePrice_skewed", data=encoded_data)
plt.figure(figsize=(20,8))

sns.regplot(x='YearBuilt', y="SalePrice_skewed", data=encoded_data)
plt.figure(figsize=(20,10))

sns.regplot(x='TotalBsmtSF', y="SalePrice_skewed", data=encoded_data, color='purple')
plt.figure(figsize=(10,8))

sns.boxplot(x="TotalBathrooms", y="SalePrice_skewed", data=encoded_data)
from math import exp

pd.set_option('display.max_columns', 500)

outliers = data[(data['TotalBathrooms']>= 5) & (data['SalePrice']<exp(13))]

outliers
plt.figure(figsize=(20,10))

sns.regplot(x='1stFlrSF', y="SalePrice_skewed", data=encoded_data, color='orange')
X = sm.add_constant(encoded_data[['GrLivArea','TotalBathrooms','OverallQual','GarageCars']])

Y = encoded_data['SalePrice_skewed']



model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 



print_model = model.summary()

print(print_model)
plt.figure(figsize=(30,10))

sns.boxplot(x=data['Neighborhood'], y=data['SalePrice'], data=data)
model = ols('SalePrice ~ Neighborhood', data = data).fit()

                

anova_result = sm.stats.anova_lm(model, typ=2)

print (anova_result)