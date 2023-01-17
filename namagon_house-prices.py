import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')

# SaleConditionをコードに変更
sale_condition = {}
for index, value in enumerate(train["SaleCondition"].unique()):
    sale_condition[value] = index
# マッピング
train["SaleCondition"] = train["SaleCondition"].map(sale_condition)
    
# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', 'MoSold', "SaleCondition"]

# Create training predictors data
train_X = train[predictor]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
# Read the test data
test = pd.read_csv('../input/test.csv')

# SaleConditionをコードに変更
sale_condition = {}
for index, value in enumerate(test["SaleCondition"].unique()):
    sale_condition[value] = index
# マッピング
test["SaleCondition"] = test["SaleCondition"].map(sale_condition)

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
# データフレームとして一旦ID, Priceで定義する
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('./'+datetime.now().strftime('%Y%m%d%H%M%S')+'_submission.csv', index=False)