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

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb



train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')
## 트레인셋 데이터셋 불러오기
train.head()
test.head()
train.info()
test.info()
train.isnull().sum()
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

pd.options.display.max_columns=100

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



def feature_engineering(data):

    data['Date'] = pd.to_datetime(data['Dates'].dt.date)

    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x: x.days)

    data['Day'] = data['Dates'].dt.day

    data['DayOfWeek'] = data['Dates'].dt.weekday

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

    data["X_Y"] = data["X"] - data["Y"]

    data["XY"] = data["X"] + data["Y"]

    data.drop(columns=['Dates','Date','Address'], inplace=True)

    return data

train = feature_engineering(train)

test = feature_engineering(test)

train.drop(columns=['Descript','Resolution'], inplace=True)
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

pd.options.display.max_columns=100

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



def feature_engineering(data):

    data['Date'] = pd.to_datetime(data['Dates'].dt.date)

    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x: x.days)

    data['Day'] = data['Dates'].dt.day

    data['DayOfWeek'] = data['Dates'].dt.weekday

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

    data["X_Y"] = data["X"] - data["Y"]

    data["XY"] = data["X"] + data["Y"]

    data.drop(columns=['Dates','Date','Address'], inplace=True)

    return data

train = feature_engineering(train)

test = feature_engineering(test)

train.drop(columns=['Descript','Resolution'], inplace=True)
train.head()
le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

X = train.drop(columns=['Category'])

y= le2.fit_transform(train['Category'])
train.head()
X.head()
train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict', ])

params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 21,

          'learning_rate': 0.4,

          'max_bin': 465,

          'num_leaves': 41,

          'verbose' : 1}



bst = lgb.train(params, train_data, 120)

predictions = bst.predict(test)



submission = pd.DataFrame(predictions, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)

submission.to_csv('LGBM_final.csv', index_label='Id')