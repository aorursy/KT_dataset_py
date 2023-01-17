import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.preprocessing import *

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print("Current directory:")
print(os.listdir("."))

# Any results you write to the current directory are saved as output.
x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')
def preprocess(x):
    x = x.copy()
    x.drop(['id', 'date'], axis=1, inplace=True)
    #x['last_build_year'] = x.apply(lambda x: max(x['renovation_year'], x['build_year']), axis=1)
    #x.drop(['renovation_year', 'build_year', 'coordinate_latitude', 'coordinate_longitude'], axis=1, inplace=True)
    return x
    
x_train = preprocess(x_train)
# y_train['price'] /= x_train['square_footage_home']
def plots(x, y, figsize=(15, 5)):
    x_names = x.columns if len(x.shape) > 1 else np.array([x.name])
    y_names = y.columns if len(y.shape) > 1 else np.array([y.name])
    cols = 6
    rows = (x_names.size * y_names.size + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(x_names.size * y_names.size):
        if rows == 1:
            if cols == 1:
                cur_ax = ax
            else:
                cur_ax = ax[i % cols]
        else:
            cur_ax = ax[i // cols][i % cols]
        x_name = x_names[i % x_names.size]
        y_name = y_names[i // x_names.size]
        cur_x = x[x_name] if x_names.size > 1 else x
        cur_y = y[y_name] if y_names.size > 1 else y
        cur_ax.set_xlabel(x_name, fontsize=10)
        cur_ax.set_ylabel(y_name, fontsize=10)
        cur_ax.scatter(cur_x, cur_y, c='b')
    
    fig.tight_layout()
    plt.show()
    
plots(x_train, y_train)
INV_SQRT_CONST = 0x5F3759DF
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=INV_SQRT_CONST)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.reshape(y_train.size)
y_test = y_test.reshape(y_test.size)
mms = MinMaxScaler().fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
classifier = GradientBoostingRegressor(learning_rate=1e-2, n_estimators=5000, criterion='mse').fit(x_train, y_train)
y_pred = classifier.predict(x_test)
rmse(y_test, y_pred)
x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')
x_test = pd.read_csv('../input/test_data.csv', index_col='index')

idx = x_test.index.values
x_train = preprocess(x_train)
x_test = preprocess(x_test)
mms = MinMaxScaler().fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
classifier = GradientBoostingRegressor(learning_rate=1e-2, n_estimators=5000, criterion='mse').fit(x_train, y_train)
y_pred = classifier.predict(x_test)
my_submission = pd.DataFrame({'index': idx, 'price': y_pred})
my_submission.to_csv('./GradientBoostingRegressor1e-2_5000.csv', index=False)