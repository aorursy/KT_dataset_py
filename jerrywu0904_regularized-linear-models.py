import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
all_data = pd.concat((

    train.loc[:, 'MSSubClass':'SaleCondition'],

    test.loc[:, 'MSSubClass':'SaleCondition'],

))
prices = pd.DataFrame({

    'price': train['SalePrice'],

    'log(1+price)': np.log1p(train['SalePrice']),

})

prices.hist()
train['SalePrice'] = np.log1p(train['SalePrice'])



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

all_data.dtypes[all_data.dtypes != "object"]
skew_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))

skew_feats = skew_feats[skew_feats > 0.75]

skew_feats = skew_feats.index