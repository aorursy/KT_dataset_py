import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.ensemble import RandomForestRegressor # library for Random Forest model

train_file_path = '../input/dataset-for-training-the-model/train.csv' # store file path
train = pd.read_csv(train_file_path) # read the training data
print(train.describe()) # print data
print(train.columns) # print name of columns 

y = train.SalePrice # set single column salePrice as target column
predic_cols = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
train_X =  train[predic_cols] # set list of columns predic_cols as predictors

forest_model = RandomForestRegressor() # create random forest model
forest_model.fit(train_X, y) # train the model using predictors and target values

test = pd.read_csv('../input/dataset-to-test-the-model-and-predict-trends/test (1).csv') # Read the test data
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predic_cols]
predic_vals = forest_model.predict(test_X) # make predictions using the model and test values
print("Making predictions for the following 5 houses:") 
print(train_X.head()) 
print("The predictions are")
print(predic_vals) # print predicted values

submission_file = pd.DataFrame({'Id': test.Id, 'SalePrice': predic_vals}) # storing the predictions in a submission file
submission_file.to_csv('submission_file.csv', index=False) # using pandas' to_csv method to write a submission file












# Any results you write to the current directory are saved as output.