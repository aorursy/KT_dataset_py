import pandas as pd
dat = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
data
cols = data.columns.tolist()
cols
pd.options.display.max_columns = None
pd.options.display.max_rows = None
data
data.shape, data_test.shape
train_features = data.drop(['SalePrice'], axis=1)
test_features = data_test
temp = pd.concat([train_features, test_features]).reset_index(drop=True)
temp.shape
total = temp.isnull().sum().sort_values(ascending=False)
percent = (temp.isnull().sum()/temp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys = ['TOTAL', 'PERCENT'])
missing_data
dr = ['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu', 'LotFrontage', 'GarageFinish', 'GarageType', 'GarageCond', 'GarageQual', 'GarageYrBlt', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType', 'MasVnrArea', 'Electrical' ]

temp = temp.drop(dr, axis=1)

total = temp.isnull().sum().sort_values(ascending=False)
percent = (temp.isnull().sum()/temp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(16)
temp.MSZoning.value_counts()
temp.describe()
temp['MSZoning'] = temp['MSZoning'].fillna('RL')
temp['BsmtFinSF1'] = temp['BsmtFinSF1'].fillna(439)
temp['BsmtFinSF2'] = temp['BsmtFinSF2'].fillna(52)
temp['BsmtUnfSF'] = temp['BsmtUnfSF'].fillna(554)
temp['TotalBsmtSF'] = temp['TotalBsmtSF'].fillna(1046)
temp['SaleType'] = temp['SaleType'].fillna('WD')

temp['BsmtHalfBath'] = temp['BsmtHalfBath'].fillna(0)
temp['BsmtFullBath'] = temp['BsmtFullBath'].fillna(0)
temp['Functional'] = temp['Functional'].fillna('Typ')
temp['Utilities'] = temp['Utilities'].fillna('AllPub')
temp['Exterior2nd'] = temp['Exterior2nd'].fillna('VinylSd')
temp['KitchenQual'] = temp['KitchenQual'].fillna('TA')
temp['GarageCars'] = temp['GarageCars'].fillna(2)
temp['Exterior1st'] = temp['Exterior1st'].fillna('VinylSd')
temp['GarageArea'] = temp['GarageArea'].fillna(472)

total = temp.isnull().sum().sort_values(ascending=False)
percent = (temp.isnull().sum()/temp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(16)


def missVals(t):
    total = t.isnull().sum().sort_values(ascending=False)
    percent = (t.isnull().sum()/t.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

print(missVals(temp))
temp = pd.get_dummies(temp).reset_index(drop=True)

cols = temp.columns
num_cols = temp._get_numeric_data().columns
num_cols
list(set(cols) - set(num_cols))

temp.shape
pd.options.display.max_columns = None
pd.options.display.max_rows = None
temp.head()
X_train = temp.iloc[:len(train_features), :]
X_test = temp.iloc[len(train_features):, :]
X_train.shape, X_test.shape
X_test.head()
X_train.head()
X_train = pd.concat([X_train, dat['SalePrice']], axis=1)
#pip install sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

traindata, testdata = train_test_split(X_train, test_size=0.2, random_state=0)

y_train = traindata['SalePrice']
X = traindata.drop(['SalePrice'], axis=1)

y_true = testdata['SalePrice']
X_minetest = testdata.drop(['SalePrice'], axis=1)

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X, y_train)

y_pred = regressor.predict(X_minetest)
y_pred
"""
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value
and the logarithm of the observed sales price.
(Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
"""
from sklearn.metrics import mean_squared_error
from math import sqrt, log

new_series = pd.Series(y_pred)

res1 = y_true.apply(lambda x : log(x))
res2 = new_series.apply(lambda x : log(x))


sqrt(mean_squared_error(res1, res2))
y_pred = regressor.predict(X_test)
y_pred
predictionsPD = pd.Series(y_pred)
submission = pd.concat([data_test['Id'], predictionsPD], axis=1)
submission=submission.rename(columns={0: "SalePrice"})
submission.to_csv("submission_file.csv", index=False)
submission