import pandas as pd

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv')

print(train.head())

train_y = train.SalePrice

predictor_columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_columns]

my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
test = pd.read_csv('../input/test.csv')

print(test.head())
test_X = test[predictor_columns]
predicted_prices = my_model.predict(test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)