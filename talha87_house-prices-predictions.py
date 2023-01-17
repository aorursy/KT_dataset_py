import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
train_y = data.SalePrice
train_X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

test_X = test_data.select_dtypes(exclude=['object'])

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
my_model.fit(train_X, train_y, verbose=False)

predictions = my_model.predict(test_X)


output = pd.DataFrame({'Id': test_data.Id,'SalePrice': predictions})
output.to_csv('submission.csv', index=False)


# Any results you write to the current directory are saved as output.
