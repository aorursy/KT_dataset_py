import pandas as pd
raw = pd.read_csv("../input/train.csv")
raw.head()
raw.columns.values
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
data = raw[features]
data.head()
data.isnull().sum()
X = data[features]
X.shape
y = raw['SalePrice']
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lr = RandomForestRegressor(n_estimators=80, max_depth=100, random_state=42)
lr.fit(X_train, y_train)
print(lr.feature_importances_)
y_pred = lr.predict(X_test)
y_pred
from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
unseen_data = pd.read_csv("../input/test.csv")
unseen_data[features].head()
unseen_data[features].shape
unseen_data[features].isnull().sum()
unseen_data[['TotalBsmtSF']] = unseen_data[['TotalBsmtSF']].fillna(value=1057)
unseen_data[features].isnull().sum()
unseen_data_pred = lr.predict(unseen_data[features])
unseen_data_pred
unseen_data_pred.shape
submission = pd.concat([unseen_data[['Id'] + features], pd.Series(unseen_data_pred, name='SalePrice')], axis=1)
submission.head()
submission.shape
result = submission[['Id', 'SalePrice']]
result.shape
result.head()
result.to_csv('result.csv', index=False)