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
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/train.csv')
import pandas_profiling
from matplotlib import pyplot as plt
data.head()
pandas_profiling.ProfileReport(data)
data = pd.read_csv('../input/train.csv')
diff = data['high'] - data['low']
data = pd.concat([data,diff],axis = 1)
data.columns.values[16] = "diff"
target = data['y']
data_test = pd.read_csv('../input/test.csv')
PredID = data_test['id']
diff_test = data_test['high'] - data_test['low']
data_test = pd.concat([data_test, diff_test],axis = 1)
data_test.columns.values[17] = 'diff'
data = data.drop('macd_signal',axis = 1).drop('id',axis = 1).drop('y',axis=1)
data_test = data_test.drop('macd_signal',axis = 1).drop('id',axis=1)
data = pd.get_dummies(data, columns = ['asset'])
data_test = pd.get_dummies(data_test, columns = ['asset'])
print(data.shape)
print(data_test.shape)
model = LR()
lin_reg = model.fit(data,target)
preds_linreg = lin_reg.predict(data_test)
# print(mean_squared_error(preds,y_test)**0.5)
out = pd.DataFrame(PredID)
out['expected'] = preds
out.to_csv('submission_1.csv',index = False)
data = pd.read_csv('../input/train.csv')
data = data.drop('macd_signal',axis = 1).drop('id',axis = 1)

print(mean_squared_error(preds,y_test)**0.5)
from sklearn.preprocessing import StandardScaler as SS
data = pd.read_csv('../input/train.csv')
data['diff'] = data['high'] - data['low']
data['diff*volume'] = data['volume'] * data['diff']
data = data.drop('macd_signal',axis = 1).drop('id',axis = 1)
x = data.drop('y',axis = 1)
y = data['y']
x = SS().fit(x).transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 777)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as SS
#train
data = pd.read_csv('../input/train.csv')
data['diff'] = data['high'] - data['low']
data['diff*volume'] = data['volume'] * data['diff']
target = data['y']
data = data.drop('macd_signal',axis = 1).drop('id',axis = 1).drop('y',axis = 1)

#test
data_test = pd.read_csv('../input/test.csv')
PredID = data_test['id']
data_test['diff'] = data_test['high'] - data_test['low']
data_test['diff*volume'] = data_test['volume'] * data_test['diff']
data_test = data_test.drop('macd_signal',axis = 1).drop('id',axis=1)
#model
rcr = RandomForestRegressor(n_estimators = 200)
rcr.fit(data,target)
preds_rcr = rcr.predict(data_test)
#print(mean_squared_error(preds_rcr,y_test)**0.5)
out = pd.DataFrame(PredID)
out['expected'] = preds_rcr
out.to_csv('submission_2.csv',index = False)

