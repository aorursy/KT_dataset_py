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
train = pd.read_csv("/kaggle/input/bitcoin-price-prediction/bitcoin_price_Training - Training.csv")
test = pd.read_csv("/kaggle/input/bitcoin-price-prediction/bitcoin_price_1week_Test - Test.csv")
train
test
train['Volume'] = train['Volume'].replace({'-' : np.nan})
train['Volume'] = train['Volume'].str.replace(',', '')
train['Market Cap'] = train['Market Cap'].str.replace(',', '')
test['Volume'] = test['Volume'].str.replace(',', '')
test['Market Cap'] = test['Market Cap'].str.replace(',', '')
train.dropna(axis=0, inplace=True)
train['Volume'] = train['Volume'].astype('int64')
train['Market Cap'] = train['Market Cap'].astype('int64')
test['Volume'] = test['Volume'].astype('int64')
test['Market Cap'] = test['Market Cap'].astype('int64')
train
test
train.dtypes, test.dtypes
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.fit_transform(test[cols])
train
test
x_train = train.drop('Market Cap', axis=1)
y_train = train['Market Cap']

x_test = test.drop('Market Cap', axis=1)
y_test = test['Market Cap']
x_train.drop('Date', axis=1, inplace=True)
x_test.drop('Date', axis=1, inplace=True)
import matplotlib.pyplot as plt
plt.plot(y_train)
plt.plot(y_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
model1 = SGDRegressor()
model2 = PassiveAggressiveRegressor()
model3 = MLPRegressor()
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
pred1 = model1.predict(x_test)
pred1
pred2 = model2.predict(x_test)
pred2
pred3 = model3.predict(x_test)
pred3
from sklearn.metrics import r2_score
acc1 = r2_score(y_test, pred1)
acc1*100
acc2 = r2_score(y_test, pred2)
acc2*100
acc3 = r2_score(y_test, pred3)
acc3*100
#We can increase the accuracy by specifing some parameters in the models (You can do it if you want)
#Now lets test with our oun random sample. We will use SGDRegressor because it got the highest acc
rpred = model1.predict([[0.395119, 0.280451, 0.038213, 0.051358, 0.355668]])
rpred
