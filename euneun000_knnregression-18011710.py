# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



#데이터 로더

train = pd.read_csv("/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv")

test = pd.read_csv("/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv")

train.head()
train.info()
train_y = train.pop('avgPrice')
train['date'] = pd.to_datetime(train['year'], format='%Y%m%d', errors='raise')

test['date'] = pd.to_datetime(test['year'], format='%Y%m%d', errors='raise')



train['Year'] = train['date'].dt.year 

train['Month'] = train['date'].dt.month 

test['Year'] = test['date'].dt.year 

test['Month'] = test['date'].dt.month 



train.drop('year', axis=1, inplace=True)

test.drop('year', axis=1, inplace=True)

train.drop('date', axis=1, inplace=True)

test.drop('date', axis=1, inplace=True)

train.head()
#스케일 조정

from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler()

sc.fit(train)

X_train_new = sc.transform(train)

test_x = sc.transform(test)
#데이터분할

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_new, train_y, test_size=0.3, random_state=5)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#학습

from sklearn.neighbors import KNeighborsRegressor



regressor = KNeighborsRegressor(n_neighbors = 25, weights = "uniform",

                               p=2)

regressor.fit(X_train, y_train)
X_train_pred = regressor.predict(X_train)

X_test_pred = regressor.predict(X_test)



# 결과 평가

print("RMSE_training: %.5f" % np.sqrt(((X_train_pred - y_train)**2).mean()))

print("RMSE_test: %.5f" % np.sqrt(((X_test_pred - y_test)**2).mean()))
# test 예측

test_pred = regressor.predict(test_x)

test_pred
#제출

submission = pd.read_csv("/kaggle/input/mlregression-cabbage-price/sample_submit.csv")

submission.head()
submission['Expected'] = submission['Expected'].astype(float)



for i in range(len(test_pred)):

    submission["Expected"][i] = test_pred[i]

submission['Expected'] = submission['Expected']
submission.head()
submission.to_csv('submission.csv', index=False, header=True)