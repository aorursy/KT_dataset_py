# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.shape
data.columns


data = data.replace(to_replace = {"No" : 0, "Yes" : 1})

data.head()
data.MultipleLines.value_counts()
data.InternetService.value_counts()
data = data.replace(to_replace = {'No phone service' : 0, 'No internet service' : 0})

data.head()
data.dtypes
data.OnlineSecurity.value_counts()
data.TotalCharges.value_counts()
data = data.drop(columns=['customerID'], axis=1)

data.shape
data = pd.get_dummies(data, columns=['gender', 'InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

data.head()
data.dtypes
data
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce').fillna(0).astype(np.float64)
X = data.drop(columns=['Churn'], axis=1)

y = data.Churn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(23,)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss = 'binary_crossentropy',

              metrics = ['acc'])

histry = model.fit(x_train, y_train, batch_size=24, epochs=100, validation_split=.20)
histry.history['val_acc']
acc = histry.history['acc']

val_acc = histry.history['val_acc']

loss = histry.history['loss']

val_loss = histry.history['val_loss']



plt.plot(acc, label='Acc')

plt.plot(val_acc, label='Val_Acc')

plt.legend()
plt.plot(loss, label='Loss')

plt.plot(val_loss, label='Val_Loss')

plt.legend()
from keras import backend as k

k.clear_session()
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(23,)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss = 'binary_crossentropy',

              metrics = ['acc'])

model.fit(x_train, y_train, batch_size=24, epochs=15, validation_split=.20)

results = model.evaluate(x_test, y_test)

results