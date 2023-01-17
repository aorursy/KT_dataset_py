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
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
train_y=train.SalePrice
predictor_cols=['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_x=train[predictor_cols]
model=RandomForestRegressor()
model.fit(train_x,train_y)
test=pd.read_csv('../input/test.csv')
test_x=test[predictor_cols]
predicted=model.predict(test_x)
print(predicted)
my_submission=pd.DataFrame({'Id': test.Id, 'SalePrice': predicted})
my_submission.to_csv('submission.csv',index=False)

