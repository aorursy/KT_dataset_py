import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]
test_X = test[predictor_cols]
#combine the training and testing data
combine = [train, test]
cols_with_missing = [col for col in train_X.columns 
                                 if train_X[col].isnull().any()]
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_test_X  = test_X.drop(cols_with_missing, axis=1)
print(reduced_train_X.columns.values)




#random forest
my_model = RandomForestRegressor()
my_model.fit(reduced_train_X, train_y)

# Treat the test data in the same way as training data. In this case, pull same columns.


# Use the model to make predictions
predicted_prices = my_model.predict(reduced_test_X)

# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

acc_my_model = round(my_model.score(train_X, train_y) * 100, 2)
acc_my_model
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)