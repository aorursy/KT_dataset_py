import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

import numpy as np
X = pd.read_csv("../input/train.csv") 

X_test_full = pd.read_csv("../input/test.csv") 



combined = [X, X_test_full]
X.head()
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)
X.isnull().sum().sort_values(ascending=False).head(20)
for dataset in combined:

    dataset['Total_Bath'] = (dataset['FullBath'] + (0.5 * dataset['FullBath']) + dataset['BsmtFullBath'] + (0.5 * dataset['BsmtHalfBath']))

    dataset['YrBltAndRemod'] = dataset['YearBuilt'] + dataset['YearRemodAdd']

    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

    dataset['Total_sqr_footage'] = (dataset['BsmtFinSF1'] + dataset['BsmtFinSF2'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'])

    dataset['Total_porch_sf'] = (dataset['OpenPorchSF'] + dataset['3SsnPorch'] + dataset['EnclosedPorch'] + dataset['ScreenPorch'] + dataset['WoodDeckSF'])
for dataset in combined:

    dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0], inplace=True)

    dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mode()[0], inplace=True)

    dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0], inplace=True)

    dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0], inplace=True)

    dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0], inplace=True)

    dataset['Electrical'].fillna(dataset['Electrical'].mode()[0], inplace=True)

    dataset['GarageType'].fillna(method='bfill', inplace=True)

    dataset['FireplaceQu'].fillna(method='bfill', inplace=True)

    dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0], inplace=True)

    dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0], inplace=True)
features = ['Utilities', 'Street', 'PoolQC']

for dataset in combined:

    dataset.drop(features, axis=1, inplace=True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and X_train_full[cname].dtype == "object"]

low_cardinality_cols
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

numeric_cols
my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
model1 = XGBRegressor(n_estimators=500, learning_rate=0.05)

model2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)



models = [model1, model2]
def housing_price(model, X_tr = X_train, X_ts = X_valid, Y_tr = y_train, Y_ts = y_valid):

    model.fit(X_tr, Y_tr)

    predictions = model.predict(X_ts)

    mae = mean_absolute_error(predictions, Y_ts)

    print("Mean Absolute Error:" , mae)
for i in range(len(models)):

    housing_price(models[i])
best_model = model2

best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.Id, 'SalePrice': preds})

output.to_csv('submission.csv', index=False)