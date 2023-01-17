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
train = pd.read_csv('../input/train.csv', index_col = 'Id')

train.head()
numeric_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

               '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr', 

               'TotRmsAbvGrd', 'GarageCars','GarageArea', 'WoodDeckSF','OpenPorchSF','EnclosedPorch', 

               '3SsnPorch','ScreenPorch','PoolArea', 'MiscVal']



standard_cols = []



for col in numeric_cols:

    mean = np.mean(train[col])

    sd = np.std(train[col])

    new_col = (train[col] - mean) / sd

    standard_cols.append('Standard' + col)

    train['Standard' + col] = new_col

    

train[standard_cols]= train[standard_cols].fillna(0)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder



preproc = Pipeline([

    (),

    (),

    ()

])

for col in train.columns:

    if train[col].dtype in ['int64', 'float64']:

        #train[col].corr(col['SalePrice'])

        print(str(col), train[col].corr(train['SalePrice']))