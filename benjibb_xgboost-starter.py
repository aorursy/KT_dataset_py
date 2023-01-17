import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
data = pd.read_csv('../input/train.csv')
# if no saleprice, drop the row
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
my_model = XGBRegressor(n_estimators = 125, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#def get_mae(n_estimators):
#    my_model = XGBRegressor(n_estimators = n_estimators, learning_rate=0.05)
#    my_model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(test_X, test_y)], verbose=False)
#    predictions = my_model.predict(test_X)
#    return mean_absolute_error(predictions, test_y)
#import numpy as np
#for i in range(180, 250, 10):
#    print ("n_estimators = %d, mae = %s" % (i, get_mae(i)))
my_model = XGBRegressor(n_estimators = 190, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
rtest = pd.read_csv('../input/test.csv')
rtest2 = pd.read_csv('../input/test.csv')
rtest = rtest.select_dtypes(exclude=['object'])
rtest = my_imputer.fit_transform(rtest)
predicted_prices = my_model.predict(rtest)

print(predicted_prices)
my_submission = pd.DataFrame({'Id': rtest2.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
