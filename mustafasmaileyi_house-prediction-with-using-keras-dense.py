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
os.getcwd()
sample_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_sub.head()
train_data.head()
test_data.head()
for i in train_data.columns:

    print(i, train_data[i].dtype)
for x in train_data.columns:

    if (train_data[x].dtype == 'object') or (train_data[x].dtype == 'O'):

        train_data.drop(columns=[x],inplace=True)
for i in train_data.columns:

    print(i, train_data[i].dtype)
train_data.head()
train_data.dropna(inplace=True)
train_data_id = train_data['Id']

train_data.drop('Id',axis=1,inplace=True)

y = train_data['SalePrice']

X = train_data.drop('SalePrice', axis=1) 
y
X.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
x_train.shape
model = Sequential()

model.add(Dense(28, activation='relu'))

model.add(Dense(14, activation='relu'))

#model.add(Dense(7, activation='relu'))

          

model.add(Dense(1))

          

model.compile(optimizer='adam',loss='mse')
#model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test),epochs=250)

model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test),epochs=133)
loss_data = pd.DataFrame(model.history.history)

loss_data.head()
loss_data.plot()
from sklearn.metrics import mean_squared_error, mean_absolute_error



tahminDizisi = model.predict(x_test)
tahminDizisi
mean_absolute_error(y_test, tahminDizisi)
from matplotlib import pyplot as plt

plt.scatter(y_test, tahminDizisi)

plt.plot(tahminDizisi, tahminDizisi, 'r-')

plt.show()
for x in test_data.columns:

    if (test_data[x].dtype == 'object') or (test_data[x].dtype == 'O'):

        test_data.drop(columns=[x],inplace=True)

        

test_data.head()
test_data.dropna(inplace=True)
ev_1 = pd.DataFrame(test_data.iloc[1,1:])

ev_1 = scaler.transform(ev_1.values.reshape(-1,36))
model.predict(ev_1)
test_data_id = test_data['Id']

test_data.drop(columns=['Id'],inplace=True)
test_data_predict = test_data.copy()

test_data_predict = scaler.transform(test_data_predict.values.reshape(-1,36))
submission=model.predict(test_data_predict)

submission.shape
submission=pd.DataFrame(submission)
submission
from matplotlib import pyplot as plt

plt.scatter(sample_sub.iloc[:submission.shape[0],1].values, submission.iloc[:].values)

plt.plot(submission.iloc[:].values, submission.iloc[:].values, 'r-')

plt.show()
sample_sub.shape
mean_absolute_error(sample_sub.iloc[:submission.shape[0],1].values,submission.iloc[:].values)
submission.to_csv('my_submission.csv')