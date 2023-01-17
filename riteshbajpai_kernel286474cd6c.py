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
housedata = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

housedata.head()
housedata.info()
housedata.describe()
housedata = housedata.fillna(housedata.mean())

housedata.head(10)
housedata = pd.get_dummies(housedata)

housedata.head(3)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# split data in dependent(x) and independent(y) variable

x = housedata.drop(['SalePrice'], axis=1)

y = housedata['SalePrice']
# Feature scaling

scaler.fit_transform(x)
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(x, y)
from sklearn.metrics import mean_squared_error, r2_score
# Check Model score

model.score(x, y)
# Check error

mean_squared_error(y, model.predict(x))