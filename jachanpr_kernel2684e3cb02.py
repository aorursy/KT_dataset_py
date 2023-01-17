# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/home_data.csv")
size = df.shape[0]
train_size = int(size * .7)
test_size = int(size * .2)
val_size = int(size * .1)
train_size, test_size, val_size
perm_idx = np.random.permutation(size)
rdf = df.iloc[perm_idx, : ]
perm_idx
rdf.head()
rdf.loc[:,"sqft_lotXfloors"] = rdf.loc[:,"sqft_lot"] * rdf.loc[:,"floors"]
rdf.head()
train_set = rdf.iloc[0:train_size, :]
test_set = rdf.iloc[train_size:train_size + test_size, :]
val_set = rdf.iloc[train_size + test_size: train_size + test_size + val_size, :]
y_train = train_set.pop("price")
X_train = train_set[["sqft_lotXfloors"]]

y_test = test_set.pop("price")
X_test = test_set[["sqft_lotXfloors"]]

y_val = val_set.pop("price")
X_val = val_set[["sqft_lotXfloors"]]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.coef_, lin_reg.intercept_
def f(x):
    return x * lin_reg.coef_ + lin_reg.intercept_
y_train_pred = lin_reg.predict(X_train)
y_val_pred = lin_reg.predict(X_val)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_val, y_val_pred)
plt.scatter(X_train, y_train)
plt.plot(X_train, f(X_train), "r")