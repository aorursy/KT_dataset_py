# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
train_data.columns
from sklearn.ensemble import RandomForestRegressor

predictors = ["LotArea","TotRmsAbvGrd", "OverallQual","YearBuilt"]
X_train = train_data[predictors]
y_train = train_data["SalePrice"]

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)

test_data = pd.read_csv('../input/test.csv')
X_test = test_data[predictors]
predicted_y = forest_model.predict(X_test)
submission_data = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_y})
submission_data.to_csv('price_prediction_submission.csv',index=False)