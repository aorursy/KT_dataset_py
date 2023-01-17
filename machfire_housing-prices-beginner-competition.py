# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor



import os

print(os.listdir("../input"))

test_set = pd.read_csv("../input/test.csv")

train_set = pd.read_csv("../input/train.csv")
train_set.head()
print(test_set.shape)

print(train_set.shape)
train_outcome_variables = train_set["SalePrice"]



predictor_variables = ["LotArea","OverallQual","OverallCond","BedroomAbvGr"]

train_predictor_variables = train_set[predictor_variables]



model = RandomForestRegressor()

model.fit(train_predictor_variables, train_outcome_variables)
test_predictor_values = test_set[predictor_variables]

test_prices = model.predict(test_predictor_values)

print(test_prices)
my_submission = pd.DataFrame({'Id':test_set['Id'], 'SalePrice':test_prices})



my_submission.to_csv("submission.csv", index=False)