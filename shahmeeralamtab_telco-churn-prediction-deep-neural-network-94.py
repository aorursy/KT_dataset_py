import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

telcom = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

telcom.head()
print ("Rows     : " ,telcom.shape[0])

print ("Columns  : " ,telcom.shape[1])

print ("\nMissing values :  ", telcom.isnull().sum().sum())

print ("\nUnique values :  \n",telcom.nunique())
import math

telcom['MonthlyCharges'] = telcom['MonthlyCharges'].apply(lambda x: math.floor(x/20))

telcom['tenure'] = telcom['tenure'].apply(lambda x: math.floor(x/10))



telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors='coerce')

telcom['TotalCharges'] = telcom['TotalCharges'].fillna(np.mean(telcom['TotalCharges']))

telcom['TotalCharges'] = telcom['TotalCharges'].apply(lambda x: math.floor(x/1000))
telcom.head()
import matplotlib.pyplot as plt

x = np.arange(1,7)

y = telcom.groupby('MonthlyCharges')['customerID'].nunique()



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(x, y)

plt.show()
x = np.arange(1,9)

y = telcom.groupby('tenure')['customerID'].nunique()



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(x, y)

plt.show()
x = np.arange(telcom.TotalCharges.nunique())

y = telcom.groupby('TotalCharges')['customerID'].nunique()



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(x, y)

plt.show()
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



telcom = telcom.apply(lambda col: le.fit_transform(col))

telcom.head(3)
telcom = telcom.drop("customerID", axis=1)

y = telcom['Churn']

telcom = telcom.drop("Churn", axis=1)
onehotencoder = OneHotEncoder(categories = 'auto')

telcom = onehotencoder.fit_transform(telcom).toarray()

X = pd.DataFrame(telcom)

X.head(3)
from keras.models import Sequential

from keras.layers import Dense





model = Sequential()

model.add(Dense(64, input_dim=66, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=10)
