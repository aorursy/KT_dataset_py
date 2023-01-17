import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_set = test.select_dtypes(exclude=['object'])
# print(test_set.columns)
# print(data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object']).columns)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X, y)
test_prediction = my_model.predict(test_set)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_prediction})
my_submission.to_csv('submission.csv', index=False)