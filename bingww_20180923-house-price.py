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
train = pd.read_csv('../input/train.csv')
train.columns
train.head()
columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[columns]
train_y = train['SalePrice']
model = RandomForestRegressor(random_state=1)
model.fit(train_X,train_y)
test = pd.read_csv('../input/test.csv')
test_X = test[columns]
preds_price = model.predict(test_X)
print(preds_price)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': preds_price})
submission.to_csv('submission.csv',index=False)
!ls
