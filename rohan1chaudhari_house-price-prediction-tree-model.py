# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_path = "../input/train.csv"

training_data = pd.read_csv(train_path)

print(training_data.describe())
important_var = ["OverallQual","LotArea","OverallCond","YearBuilt","YearRemodAdd","YrSold","GarageCars","SalePrice"]

important_data = training_data[important_var]

print(important_data.head())

filtered_train_data = important_data.dropna(axis=0)

print(filtered_train_data)
trainingX = filtered_train_data[["OverallQual","LotArea","OverallCond","YearBuilt","YearRemodAdd","YrSold","GarageCars"]]

trainingY = filtered_train_data.SalePrice
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(trainingX,trainingY)

test_path = "../input/test.csv"

test_data = pd.read_csv(test_path)

imp_test_data = test_data[["Id","OverallQual","LotArea","OverallCond","YearBuilt","YearRemodAdd","YrSold","GarageCars"]]

filtered_test_data = imp_test_data.dropna(axis=0)

print(filtered_test_data)

predictions = model.predict(filtered_test_data[["OverallQual","LotArea","OverallCond","YearBuilt","YearRemodAdd","YrSold","GarageCars"]])







test_id = filtered_test_data["Id"]

results = pd.DataFrame({

    "Id" : test_id,

    "SalePrice" : pd.Series(predictions),

})

results.to_csv("results.csv",index=False)



print(check_output(["ls"]).decode("utf8"))