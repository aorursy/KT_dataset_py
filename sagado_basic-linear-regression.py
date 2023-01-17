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
# Load the training data

train_data = pd.read_csv("../input/train.csv")
# Load test data

test_data = pd.read_csv("../input/test.csv")
from sklearn.model_selection import cross_val_predict

from sklearn import linear_model
features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

            '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

            'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
# Train linear model on selected features

lr = linear_model.LinearRegression()

lr.fit(train_data[features], train_data['SalePrice'])

# Predict Sale Price for test data 

# (few NaN in test data, just filled with zero)

predicted = lr.predict(test_data[features].fillna(0))
[(i, price) for (i, price) in enumerate(predicted) if price<100]