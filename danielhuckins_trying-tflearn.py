# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn
data = pd.read_csv('../input/train.csv')

data = data.drop(['Id'], axis=1)

sales = data['SalePrice']

data = data.drop(['SalePrice'], axis=1)

data = data.dropna()

# categorical columns

cat_cols = [

    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',

    'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

    'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',

    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',

    'CentralAir', 'Electrical', 'KitchenQual', 'Functional',

    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

    'SaleType', 'SaleCondition'

]
from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    data[col] = le.fit_transform(data[col])

    encoders[col] = le
data['Alley']
net = tflearn.input_data(shape=[None, 80])

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 32)

net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_verbose=1)

model.fit(data, sales, show_metric=True)
model.inputs