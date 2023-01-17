# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train_y = train.SalePrice
predict_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predict_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_X = test[predict_cols]
predicted_prices = my_model.predict(test_X)
predicted_prices
submission = pd.DataFrame({'Id':test.Id, 'SalePrice':predicted_prices})
submission.to_csv('submission.csv', index=False)