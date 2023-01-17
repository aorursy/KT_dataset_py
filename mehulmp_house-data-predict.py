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
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('../input/train.csv')
train_data.describe()

y = train_data['SalePrice']
features = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

X = train_data[features]

model = RandomForestRegressor()
model.fit(X, y)
test_data = pd.read_csv('../input/test.csv')
X_test = test_data[features]
prediction = model.predict(X_test)
print(prediction)
submit = pd.DataFrame({'Id': test_data.Id, 'SalePrice': prediction})

submit.to_csv('submission.csv', index=False)
