# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
train = pd.read_csv('../input/housing-prices-competition-for-kaggle-learn-users/train.csv')
test = pd.read_csv('../input/housing-prices-competition-for-kaggle-learn-users/test.csv')
from sklearn.model_selection import train_test_split

train.info()
selected = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'LowQualFinSF',
          'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageArea', 'PoolArea', 'MiscVal']
X = train[selected]
X.fillna(value=0, inplace = True)
y = train['SalePrice']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
train_X
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


model = RandomForestRegressor(random_state = 1)
model.fit(train_X, train_y)
prediction = model.predict(val_X)
MAE = mean_absolute_error(val_y, prediction)
compare = pd.DataFrame(data={'prediction':prediction, 'actual':val_y, 'MAE':MAE})
compare.describe()
test.info()
test_X = test[selected]
test_X.fillna(value=0, inplace=True)
test_pred = model.predict(test_X)
output = pd.DataFrame(data = {'Id':test.Id,'SalePrice':test_pred})
#pd.concat([compare, test_df], ignore_index = True, axis = 1)
#test_df.describe()
output.to_csv('submission.csv', index = False)
output.info()