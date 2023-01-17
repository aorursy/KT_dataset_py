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
file_path='/kaggle/input/home-data-for-ml-course/train.csv'

test_file='/kaggle/input/home-data-for-ml-course/test.csv'

data = pd.read_csv(file_path)

test_data = pd.read_csv(test_file)

test_data.describe()
train_y = data['SalePrice']

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

train_X = data[ features ]

test_X = test_data[ features ]

#test_y = test_data['SalePrice']

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)



model.fit(train_X,train_y)



p = model.predict(test_X)

print(p)

#from sklearn.metrics import mean_absolute_error

#mae = mean_absolute_error(p,test_y)