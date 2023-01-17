import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso

from sklearn.preprocessing import MinMaxScaler
df  = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
corr = df.corr()['SalePrice'].sort_values(ascending=False).head(20).to_frame()
plt.figure(figsize=(10,5))

sns.heatmap(corr)
plt.figure(figsize=(15,10))

sns.jointplot(x='OverallQual', y='SalePrice', data=df)
plt.figure(figsize=(15,10))

sns.jointplot(x='GrLivArea', y='SalePrice', data=df)
plt.figure(figsize=(15,10))

sns.countplot(x='Neighborhood', data=df, order=df['Neighborhood'].value_counts().index)

plt.xticks(rotation=60)
Neighborhood = dict(zip(df['Neighborhood'].unique().tolist(), range(len(df['Neighborhood'].unique().tolist()))))

df.replace({'Neighborhood': Neighborhood}, inplace=True)

plt.figure(figsize=(15,10))

sns.barplot(x='Neighborhood', y='SalePrice', data=df)

plt.xlabel('Neighborhood')

plt.xticks([*range(0, len(Neighborhood))], Neighborhood, rotation=60)
HouseStyle = dict(zip(df['HouseStyle'].unique().tolist(), range(len(df['HouseStyle'].unique().tolist()))))

df.replace({'HouseStyle': HouseStyle}, inplace=True)

plt.figure(figsize=(15,10))

sns.barplot(x='HouseStyle', y='SalePrice', data=df)

plt.xlabel('HouseStyle')

plt.xticks([*range(0, len(HouseStyle))], HouseStyle, rotation=60)
BsmtFinType1 = dict(zip(df['BsmtFinType1'].unique().tolist(), range(len(df['BsmtFinType1'].unique().tolist()))))

df.replace({'BsmtFinType1': BsmtFinType1}, inplace=True)

plt.figure(figsize=(15,10))

sns.barplot(x='BsmtFinType1', y='SalePrice', data=df)

plt.xlabel('BsmtFinType1')

plt.xticks([*range(0, len(BsmtFinType1))], BsmtFinType1, rotation=60)
BldgType = dict(zip(df['BldgType'].unique().tolist(), range(len(df['BldgType'].unique().tolist()))))

df.replace({'BldgType': BldgType}, inplace=True)

plt.figure(figsize=(15,10))

sns.barplot(x='BldgType', y='SalePrice', data=df)

plt.xlabel('BldgType')

plt.xticks([*range(0, len(BldgType))], BldgType, rotation=60)
plt.figure(figsize=(15,10))

df.isnull().mean().sort_values(ascending=False).plot()
df['FireplaceQu'] = df['FireplaceQu'].fillna(value='NF')

df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace=True)

df['LotFrontage'] = df['LotFrontage'].fillna(value=df['LotFrontage'].mean())

df['GarageType'] = df['GarageType'].fillna(value='NoGar')

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value=df['GarageYrBlt'].mean())

df['GarageQual'] = df['GarageQual'].fillna(value='NoGar')

df['GarageFinish'] = df['GarageFinish'].fillna(value='NoGar')

df['GarageCond'] = df['GarageCond'].fillna(value='NoGar')

df['BsmtFinType2'] = df['BsmtFinType2'].fillna(value='NoBasement')

df['BsmtExposure'] = df['BsmtExposure'].fillna(value='NoBasement')

df['BsmtQual'] = df['BsmtQual'].fillna(value='NoBasement')

df['BsmtCond'] = df['BsmtCond'].fillna(value='NoBasement')

df['MasVnrType'] = df['MasVnrType'].fillna(value='None')

df['MasVnrArea'] = df['MasVnrArea'].fillna(value=0.0)



Electrical = dict(zip(df['Electrical'].unique().tolist(), range(len(df['Electrical'].unique().tolist()))))

df.replace({'Electrical': Electrical}, inplace=True)

df['Electrical'] = df['Electrical'].fillna(value=0)
df.isnull().mean().sort_values(ascending=False)
for column in df.columns:

    if(df[column].dtype == 'object'):

        df.replace({column: dict(zip(df[column].unique().tolist(), range(len(df[column].unique().tolist()))))}, inplace=True)

df.head()
df['totalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] + df['GarageArea']

df['Bathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5

df['Year average'] = (df['YearRemodAdd'] + df['YearBuilt']) / 2
new_corr = pd.DataFrame({'Feature Name': ['Total Area', 'Bathrooms', 'Year Average'], 

                         'Corr': [df['totalArea'].corr(df['SalePrice']), df['Bathrooms'].corr(df['SalePrice']), df['Year average'].corr(df['SalePrice'])]})

new_corr
y = df['SalePrice']

X = df.drop(columns='SalePrice')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler= MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
models = {

    'linear regression': LinearRegression(),

    'gradient boosting regressor': GradientBoostingRegressor(n_estimators=2000, max_depth=1),

    'lasso regression': Lasso()

}
score_df = pd.DataFrame({'Model': [], 'Accuracy': []})



for key, value in models.items():

    model = value

    model.fit(X_train,y_train)

    score = model.score(X_test, y_test)

    

    score_df = score_df.append({

        'Model': key,

        'Accuracy': score * 100

    }, ignore_index=True)
score_df