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
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression as LR
# import numpy as np
# import pandas as pd
# import pandas_profiling
# from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error

# data = pd.read_csv('../input/train.csv')
# data_test = pd.read_csv('../input/test.csv')
# PredID = data_test['id']
# target = data['y']
# data = pd.read_csv('../input/train.csv')
# data_test = pd.read_csv('../input/test.csv')
# PredID = data_test['id']
# target = data['y']
# data.drop('id',axis = 1).drop('asset',axis = 1).drop('macd_signal',axis = 1).drop('di_plus',axis = 1).drop('di_minus',axis=1).drop('rsi',axis = 1).drop('cci',axis = 1).drop('adl',axis=1)
# data['diff'] = data['high'] - data['low']
# data['diff*volume'] = data['volume'] * data['diff']
# data['open*close'] = data['open'] * data['close']
# data['open*high'] = data['open'] * data['high']
# data['close*high'] = data['close'] * data['high']
# data['open*macd_hist'] = data['open'] * data['macd_hist']
# data['close*macd_hist'] = data['close'] * data['macd_hist']
# data['high*macd_hist'] = data['high'] * data['macd_hist']

# # data_train = data.drop('y',axis = 1)

# data_test.drop('id',axis = 1).drop('asset',axis = 1).drop('macd_signal',axis = 1).drop('di_plus',axis = 1).drop('di_minus',axis=1).drop('rsi',axis = 1).drop('cci',axis = 1).drop('adl',axis=1)
# data_test['diff'] = data_test['high'] - data_test['low']
# data_test['diff*volume'] = data_test['volume'] * data_test['diff']
# data_test['open*close'] = data_test['open'] * data_test['close']
# data_test['open*high'] = data_test['open'] * data_test['high']
# data_test['close*high'] = data_test['close'] * data_test['high']
# data_test['open*macd_hist'] = data_test['open'] * data_test['macd_hist']
# data_test['close*macd_hist'] = data_test['close'] * data_test['macd_hist']
# data_test['high*macd_hist'] = data_test['high'] * data_test['macd_hist']
data = data.drop('y',axis = 1)
# x_train,x_test,y_train, y_test = train_test_split(data_train,target,test_size = 0.2, random_state = 777)
# LinReg = LR()
# LinReg.fit(data,target)
# preds_linreg = LinReg.predict(data_test)
# from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.metrics import mean_squared_error
# x_train, x_test, y_train, y_test = train_test_split(data,target, test_size = 0.2, random_state = 777)
# model = RFR(max_depth = 7, min_samples_split = 3, n_estimators = 250)
# model.fit(x_train, y_train)
# preds_rfc = model.predict(x_test)
# print(mean_squared_error(preds_rfc, y_test)**0.5)

from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler as SS
import pandas as pd
data = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
PredID = data_test['id']

data = pd.get_dummies(data, columns=['asset'])
data.drop('id',axis = 1).drop('macd_signal',axis = 1)
data['diff'] = data['high'] - data['low']
data['diff_price'] = data['open'] - data['close']
data['diff*volume'] = data['volume'] * data['diff']
data['open*close'] = data['open'] * data['close']
data['open*high'] = data['open'] * data['high']
data['close*high'] = data['close'] * data['high']
data['open*macd_hist'] = data['open'] * data['macd_hist']
data['close*macd_hist'] = data['close'] * data['macd_hist']
data['high*macd_hist'] = data['high'] * data['macd_hist']

data_test = pd.get_dummies(data_test, columns=['asset'])
data_test.drop('id',axis = 1).drop('macd_signal',axis = 1)
data_test['diff'] = data_test['high'] - data_test['low']
data_test['diff_price'] = data_test['open'] - data_test['close']
data_test['diff*volume'] = data_test['volume'] * data_test['diff']
data_test['open*close'] = data_test['open'] * data_test['close']
data_test['open*high'] = data_test['open'] * data_test['high']
data_test['close*high'] = data_test['close'] * data_test['high']
data_test['open*macd_hist'] = data_test['open'] * data_test['macd_hist']
data_test['close*macd_hist'] = data_test['close'] * data_test['macd_hist']
data_test['high*macd_hist'] = data_test['high'] * data_test['macd_hist']

target = data['y']
data = data.drop('y',axis = 1)
data = SS().fit(data).transform(data)
# x_train, x_test, y_train, y_test = train_test_split(data,target,test_size = 0.2, random_state = 777)
model = AdaBoostRegressor(dtr(max_depth = 7), n_estimators = 250)
model.fit(data,target)
y_preds_dtr = model.predict(data_test)
# print(mean_squared_error(y_test,y_preds_dtr)**0.5)
out = pd.DataFrame(PredID)
out['expected'] = y_preds_dtr
out.to_csv('submission_4.csv',index = False)

