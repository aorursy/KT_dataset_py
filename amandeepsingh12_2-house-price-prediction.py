import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import cufflinks as cf

cf.go_offline()
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.shape
test.head()
test.shape
plt.figure(figsize = (12, 8))

sns.heatmap(train.isnull(), yticklabels = False, cbar = False)
train_nan_vals = dict(train.isnull().sum())

for i, j in train_nan_vals.items():

    print(i   ,'-->',  j)
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train.drop(['Alley'], axis = 1, inplace = True)
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])



train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])



train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])



train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])



train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])



train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])

train.drop(['Id', 'PoolQC','Fence','MiscFeature'], axis = 1, inplace = True)
train.isnull().sum().any()
train.shape
train.isnull().sum().any()
test.info()
plt.figure(figsize = (12,6))

sns.heatmap(test.isnull(), cmap = 'viridis')
test_nan_vals = dict(test.isnull().sum())

for i, j in test_nan_vals.items():

    print(i   ,'--->',  j)
test['BsmtFinType2'].value_counts()
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])



test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())





test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])



test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])



test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())



test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])

test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())



test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean())

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean())



test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])



test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])



test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])



test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])



test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])



test.drop(['Id', 'Alley','PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
test.shape
test.isnull().sum().any()
corr = train.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr)
n = 10

cor_cols = corr.nlargest(n, 'SalePrice')['SalePrice'].index

cor_cols_1 = np.corrcoef(train[cor_cols].values.T)
sns.set(font_scale=1.50)

plt.figure(figsize = (16,10))

sns.heatmap(cor_cols_1, cbar=True, annot=True, yticklabels=cor_cols.values, xticklabels=cor_cols.values)
cor_cols = pd.DataFrame(cor_cols)

cor_cols.columns = ['Top correlated features']

cor_cols
overall_qua = train['OverallQual'].value_counts().reset_index().rename(columns = {'index': 'Rating', 'OverallQual': 'No. of customers'})

overall_qua
fig = px.pie(names = overall_qua['Rating'].values, values = overall_qua['No. of customers'].values, title='overall quality')

fig.show()
sns.set(style="whitegrid")

plt.figure(figsize=(15,10))

fig = sns.barplot(x = 'OverallQual', y='SalePrice', data = train)
sns.set(style="whitegrid")

plt.figure(figsize=(15,10))

fig = sns.scatterplot(x = 'GrLivArea', y='SalePrice', data = train)

train['GarageCars'].describe()
fig = px.box(train, x='GarageCars', y = 'SalePrice')

fig.show()
train['GarageArea'].describe()
fig = px.scatter(train, x = 'GarageArea', y = 'SalePrice')

fig.show()
plt.figure(figsize = (16,7))

sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = train)
train['1stFlrSF'].describe()
fig = px.scatter(train, x = '1stFlrSF', y = 'SalePrice')

fig.show()
train['FullBath'].value_counts()
plt.figure(figsize = (16,7))

sns.boxplot(x=train['FullBath'], y=train['SalePrice'])
train['TotRmsAbvGrd'].value_counts()
plt.figure(figsize=(15,10))

fig = sns.barplot(x = 'TotRmsAbvGrd', y='SalePrice', data = train)
train['YearBuilt'].value_counts().shape
plt.figure(figsize = (16,7)) 

fig = px.bar(train, x = 'YearBuilt', y = 'SalePrice')

fig.show()
dataframe = pd.concat([train, test])
dataframe.shape
df = pd.get_dummies(dataframe, drop_first=True)
df.head()
train_df = df.iloc[: 1460, :]

test_df = df.iloc[1460: , :]
X = train_df.drop('SalePrice', axis=1)

y = train_df['SalePrice']

test_df = test_df.drop('SalePrice', axis=1)
from sklearn.ensemble import RandomForestRegressor



forest_regressor = RandomForestRegressor()

forest_regressor.fit(X, y)

forest_regressor.score(X, y)
from sklearn.svm import SVC

classifier_SVC = SVC(kernel = 'linear', random_state = 0)

classifier_SVC.fit(X, y)

classifier_SVC.score(X, y)
from sklearn.naive_bayes import GaussianNB

classifier_NB = GaussianNB()

classifier_NB.fit(X, y)

classifier_NB.score(X, y)
from sklearn.tree import DecisionTreeClassifier

classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier_DTC.fit(X, y)

classifier_DTC.score(X, y)
y_pred = forest_regressor.predict(test_df)
sample_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets = pd.DataFrame({

    'Id': sample_sub['Id'],

    'SalePrice': y_pred

})
datasets.to_csv('gender_submission3.csv', index=False)



print("File Saved")