import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing.imputation import Imputer

# Read the data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice


# Create training predictors data
train.drop(['SalePrice'], axis=1)
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','MSSubClass','MSZoning','HeatingQC','CentralAir','FullBath']
train_X = train[predictor_cols]
my_imputer = Imputer()
train_X = pd.get_dummies(train_X)
train_X = my_imputer.fit_transform(train_X)


my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
train_X = pd.DataFrame(train_X)
# Read the test data
test = pd.read_csv('../input/test.csv')
test_X = test[predictor_cols]
test_X = pd.get_dummies(test_X)
test_X = my_imputer.fit_transform(test_X)
test_X = pd.DataFrame(test_X)
final_train, final_test = train_X.align(test_X,join='inner',axis=1)
final_test = final_test.fillna(0)

# Use the model to make predictions
predicted_prices = my_model.predict(final_test)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission4.csv', index=False)