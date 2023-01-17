# Imports

import numpy as np

import pandas as pd

import sklearn

from sklearn import linear_model
# Read data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Extract numeric features

all_ex = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

all_ex = all_ex.loc[:,all_ex.dtypes[all_ex.dtypes != "object"].index]



# Replace NaNs

all_ex = all_ex.fillna(all_ex.mean())
y = train.SalePrice

y.hist()
y_log = np.log(y)

y_log.hist()
# Split in train an test datasets

X_train = all_ex[:train.shape[0]]

X_test = all_ex[train.shape[0]:]
# LinearRegression



lr = linear_model.LinearRegression()

lr.fit(X_train,y_log)
p_lr=lr.predict(X_test)

p_lr = np.exp(p_lr)
out = pd.Series(p_lr,index=test.loc[:,'Id'])

out.name = 'SalePrice'

out.to_csv('out.csv', header=True, index_label='Id')