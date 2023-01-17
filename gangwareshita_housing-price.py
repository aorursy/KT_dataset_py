# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
reg=LinearRegression()
x = pd.read_csv('../input/train.csv')
train_x = x[['YrSold','MoSold' , 'LotArea' ,'BedroomAbvGr']]
train_y = x['SalePrice']
y = pd.read_csv('../input/test.csv')
test_x = y[['YrSold','MoSold' , 'LotArea' ,'BedroomAbvGr']]
id = y['Id']


# Any results you write to the current directory are saved as output.
reg=reg.fit(train_x,train_y)
pred_class=reg.predict(test_x)
df = pd.DataFrame()
df['Id'] = id
df['SalePrice'] = pred_class
df.to_csv('submission.csv', index = False)