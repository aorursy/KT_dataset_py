import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data_encoded = pd.get_dummies(data)

abs_corr = abs(data_encoded.corr()['SalePrice'])

cols = data_encoded.columns[abs_corr > 0.55].drop('SalePrice')

atts = []

for i in cols:

    index = i.find('_')

    if index > 0:

        i = i[:index]

    if i not in atts and i in data_encoded.columns:

        atts.append(i)

preprocessor = ColumnTransformer(transformers=[

    ('num', StandardScaler(), [i for i in atts if data[i].dtypes == 'int64']),

    ('cat', Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

        ('onehot', OneHotEncoder())

    ]), [i for i in atts if data[i].dtypes == 'object'])

])

X_train = preprocessor.fit_transform(data_encoded[atts])

y_train = data[['SalePrice']]

xgb = XGBRegressor(n_estimators=500, n_jobs=-1).fit(X_train, y_train)

print('Training RMSE: ' + str(sqrt(mean_squared_error(y_train, xgb.predict(X_train)))))

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

X_test = preprocessor.transform(pd.get_dummies(test)[atts])

pd.DataFrame({'Id': test.Id, 'SalePrice': xgb.predict(X_test)}).to_csv('submission.csv', index=False)