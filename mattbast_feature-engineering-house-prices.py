import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import time

import datetime



from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import skew
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)



data = train_data.append(test_data)
train_data['SalePrice'].hist()

plt.xticks(rotation=45)
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

train_data['SalePrice'].hist()
numerical_columns = data.select_dtypes(include=['int64', 'float64'])

print(numerical_columns.info())
for col in data:

    datatype = data[col].dtype 



    if datatype == np.int64 or datatype == np.float64:

        data[col] = data[col].fillna(data[col].mean())
figure, axes = plt.subplots(figsize=(12, 9))



correlation_matrix = train_data.corr()



cols = correlation_matrix.nlargest(10, 'SalePrice')['SalePrice'].index

correlation_matrix = np.corrcoef(train_data[cols].values.T)



sns.set(font_scale=1.25)

heatmap = sns.heatmap(

    correlation_matrix, 

    cbar=True, 

    annot=True, 

    square=True, 

    cmap=plt.cm.RdYlGn,

    fmt='.2f', 

    annot_kws={'size': 10}, 

    yticklabels=cols.values, 

    xticklabels=cols.values

)
figure, axes = plt.subplots(3, 3, figsize=(16, 12))



train_data.plot.scatter(x="OverallQual", y="SalePrice", ax=axes[0, 0])

train_data.plot.scatter(x="GrLivArea", y="SalePrice", ax=axes[0, 1])

train_data.plot.scatter(x="GarageCars", y="SalePrice", ax=axes[0, 2])

train_data.plot.scatter(x="GarageArea", y="SalePrice", ax=axes[1, 0])

train_data.plot.scatter(x="TotalBsmtSF", y="SalePrice", ax=axes[1, 1])

train_data.plot.scatter(x="1stFlrSF", y="SalePrice", ax=axes[1, 2])

train_data.plot.scatter(x="FullBath", y="SalePrice", ax=axes[2, 0])

train_data.plot.scatter(x="YearBuilt", y="SalePrice", ax=axes[2, 1])

train_data.plot.scatter(x="YearRemodAdd", y="SalePrice", ax=axes[2, 2])
data['GrLivArea'] = data['GrLivArea'].clip(0, 3500)

data['TotalBsmtSF'] = data['TotalBsmtSF'].clip(0, 3500)

data['1stFlrSF'] = data['1stFlrSF'].clip(0, 3500)
data['GarageArea'] = data['GarageArea'].clip(0, 1200)
data['YearBuilt'] = data['YearBuilt'].clip(1950, data['YearBuilt'].max())

data['YearRemodAdd'] = data['YearRemodAdd'].clip(1950, data['YearRemodAdd'].max())
figure, axes = plt.subplots(3, 3, figsize=(16, 12))



train_data.plot.scatter(x="BsmtFinSF1", y="SalePrice", ax=axes[0, 0])

train_data.plot.scatter(x="BsmtFinSF2", y="SalePrice", ax=axes[0, 1])

train_data.plot.scatter(x="BsmtUnfSF", y="SalePrice", ax=axes[0, 2])

train_data.plot.scatter(x="2ndFlrSF", y="SalePrice", ax=axes[1, 0])

train_data.plot.scatter(x="LowQualFinSF", y="SalePrice", ax=axes[1, 1])

train_data.plot.scatter(x="WoodDeckSF", y="SalePrice", ax=axes[1, 2])

train_data.plot.scatter(x="OpenPorchSF", y="SalePrice", ax=axes[2, 0])

train_data.plot.scatter(x="LotArea", y="SalePrice", ax=axes[2, 1])

train_data.plot.scatter(x="PoolArea", y="SalePrice", ax=axes[2, 2])
data['BsmtFinSF1'] = data['BsmtFinSF1'].clip(0, 2500)

data['OpenPorchSF'] = data['OpenPorchSF'].clip(0, 400)

data['LotArea'] = data['LotArea'].clip(0, 70000)
data = data.drop(columns=['PoolArea'])
figure, axes = plt.subplots(2, 3, figsize=(16, 8))



train_data.plot.scatter(x="TotRmsAbvGrd", y="SalePrice", ax=axes[0, 0])

train_data.plot.scatter(x="BsmtFullBath", y="SalePrice", ax=axes[0, 1])

train_data.plot.scatter(x="BsmtHalfBath", y="SalePrice", ax=axes[0, 2])

train_data.plot.scatter(x="HalfBath", y="SalePrice", ax=axes[1, 0])

train_data.plot.scatter(x="BedroomAbvGr", y="SalePrice", ax=axes[1, 1])

train_data.plot.scatter(x="KitchenAbvGr", y="SalePrice", ax=axes[1, 2])
data = data.drop(columns=[

    'KitchenAbvGr', 

    'BsmtFullBath', 

    'BsmtHalfBath'

])



data['TotRmsAbvGrd'] = data['TotRmsAbvGrd'].clip(0, 11)

data['HalfBath'] = data['HalfBath'].clip(0, 1)

data['BedroomAbvGr'] = data['BedroomAbvGr'].clip(0, 4)
figure, axes = plt.subplots(2, 2, figsize=(12, 8))



train_data.plot.scatter(x="GarageYrBlt", y="SalePrice", ax=axes[0, 0])

train_data.plot.scatter(x="MoSold", y="SalePrice", ax=axes[0, 1])

train_data.plot.scatter(x="YrSold", y="SalePrice", ax=axes[1, 0])
data['GarageYrBlt'] = data['GarageYrBlt'].clip(1950, data['GarageYrBlt'].max())
train_data['DateSold'] = '01/' + train_data['MoSold'].astype(str) + '/' + train_data['YrSold'].astype(str)

train_data['DateSold'] = pd.to_datetime(train_data['DateSold'], format='%d/%m/%Y')

train_data['DateSold'] = (train_data['DateSold'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')



train_data.plot.scatter(x="DateSold", y="SalePrice")
data['DateSold'] = '01/' + data['MoSold'].astype(str) + '/' + data['YrSold'].astype(str)

data['DateSold'] = pd.to_datetime(data['DateSold'], format='%d/%m/%Y')

data['DateSold'] = (data['DateSold'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')



data = data.drop(columns=['MoSold', 'YrSold'])
figure, axes = plt.subplots(3, 3, figsize=(16, 12))



train_data.plot.scatter(x="Fireplaces", y="SalePrice", ax=axes[0, 0])

train_data.plot.scatter(x="MasVnrArea", y="SalePrice", ax=axes[0, 1])

train_data.plot.scatter(x="MSSubClass", y="SalePrice", ax=axes[0, 2])

train_data.plot.scatter(x="OverallCond", y="SalePrice", ax=axes[1, 0])

train_data.plot.scatter(x="EnclosedPorch", y="SalePrice", ax=axes[1, 1])

train_data.plot.scatter(x="3SsnPorch", y="SalePrice", ax=axes[1, 2])

train_data.plot.scatter(x="ScreenPorch", y="SalePrice", ax=axes[2, 0])

train_data.plot.scatter(x="MiscVal", y="SalePrice", ax=axes[2, 1])
data['Fireplaces'] = data['Fireplaces'].clip(0, 2)

data['MasVnrArea'] = data['MasVnrArea'].clip(0, 1000)

data['EnclosedPorch'] = data['EnclosedPorch'].clip(0, 400)

data['ScreenPorch'] = data['ScreenPorch'].clip(0, 300)



data = data.drop(columns=[

    'MSSubClass', 

    '3SsnPorch', 

    'MiscVal'

])
numerical_columns = data.select_dtypes(include=['int64', 'float64'])

skewness_of_feats = numerical_columns.apply(lambda x: skew(x)).sort_values(ascending=False)

print(skewness_of_feats)
data['LowQualFinSF'] = np.log1p(data['LowQualFinSF'])

data['LotArea'] = np.log1p(data['LotArea'])

data['BsmtFinSF2'] = np.log1p(data['BsmtFinSF2'])

data['ScreenPorch'] = np.log1p(data['ScreenPorch'])

data['EnclosedPorch'] = np.log1p(data['EnclosedPorch'])

data['MasVnrArea'] = np.log1p(data['MasVnrArea'])

data['OpenPorchSF'] = np.log1p(data['OpenPorchSF'])

data['WoodDeckSF'] = np.log1p(data['WoodDeckSF'])

data['LotFrontage'] = np.log1p(data['LotFrontage'])

data['1stFlrSF'] = np.log1p(data['1stFlrSF'])
categorical_columns = data.select_dtypes(include='object')

print(categorical_columns.info())
data = data.drop(columns=[

    'PoolQC', 

    'Fence', 

    'MiscFeature', 

    'Alley', 

    'FireplaceQu'

])
for col in data:

    datatype = data[col].dtype 



    if datatype == 'object':

        data[col] = data[col].fillna('None')
labels = train_data.pop('SalePrice')

data = data.drop(columns=['SalePrice'])
features = pd.get_dummies(data)
train_features = features[:train_data.shape[0]]

test_features = features[train_data.shape[0]:]
model = LinearRegression(fit_intercept=False, copy_X=True)



model.fit(train_features, labels)
predictions = model.predict(train_features)



mean_squared_error(

    np.log(np.expm1(labels)),

    np.log(np.expm1(predictions)),

    squared=False

)
delta = np.expm1(predictions) - np.expm1(labels)

plt.hist(delta, bins=20)



plt.xticks(rotation=45)
plt.scatter(np.expm1(predictions), np.expm1(labels))



plt.xlabel('predictions')

plt.xticks(rotation=45)

plt.ylabel('labels')
ridge_model = Ridge(alpha=0.5, fit_intercept=False, copy_X=True)



ridge_model.fit(train_features, labels)
ridge_predictions = ridge_model.predict(train_features)



mean_squared_error(

    np.log(np.expm1(labels)),

    np.log(np.expm1(ridge_predictions)),

    squared=False

)
lasso_model = Lasso(alpha=0.5, fit_intercept=False, copy_X=True)



lasso_model.fit(train_features, labels)
lasso_predictions = lasso_model.predict(train_features)



mean_squared_error(

    np.log(np.expm1(labels)),

    np.log(np.expm1(lasso_predictions)),

    squared=False

)
elastic_model = ElasticNet(alpha=0.5, fit_intercept=False, copy_X=True)



elastic_model.fit(train_features, labels)
elastic_predictions = elastic_model.predict(train_features)



mean_squared_error(

    np.log(np.expm1(labels)),

    np.log(np.expm1(elastic_predictions)),

    squared=False

)
predictions = ridge_model.predict(test_features)
submission = pd.DataFrame(data={'SalePrice': np.expm1(predictions)}, index=test_data.index)



submission.index = submission.index.rename('Id')



submission.to_csv('submission_file.csv')
submission.head()