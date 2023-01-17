import numpy as np

import pandas as pd
data = pd.read_csv('../input/Churn_Modelling.csv')
data.head()
data.info()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
scaler_credit_score = StandardScaler()



data['CreditScore'] = data['CreditScore'].astype('float64')

data['credit_score'] = scaler_credit_score.fit_transform(data.CreditScore.values.reshape(-1, 1))
encoder_geography = LabelEncoder()



data['geography'] = encoder_geography.fit_transform(data.Geography)
encoder_gender = LabelEncoder()



data['gender'] = encoder_gender.fit_transform(data.Gender)
scaler_age = MinMaxScaler()



data['Age'] = data['Age'].astype('float64')

data['age'] = scaler_age.fit_transform(data.Age.values.reshape(-1,1))
scaler_tenure = MinMaxScaler()



data['Tenure'] = data['Tenure'].astype('float64')

data['tenure'] = scaler_tenure.fit_transform(data.Tenure.values.reshape(-1,1))
scaler_balance = StandardScaler()



data['balance'] = scaler_balance.fit_transform(data.Balance.values.reshape(-1, 1))
scaler_salary = StandardScaler()



data['salary'] = scaler_salary.fit_transform(data.EstimatedSalary.values.reshape(-1, 1))
data.head()
data.drop(['RowNumber', 

           'CustomerId', 

           'Surname', 

           'CreditScore', 

           'Geography', 

           'Gender', 

           'Age', 

           'Tenure', 

           'Balance', 

           'EstimatedSalary'], axis=1, inplace=True)
data = data.rename(columns={'NumOfProducts': 'product_num', 

                            'HasCrCard': 'credit_card', 

                            'IsActiveMember': 'active_member', 

                            'Exited': 'exited'})
exited = data['exited']

data.drop(['exited'], axis=1, inplace=True)

data.insert(0, 'exited', exited)
data.head()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))

sns.heatmap(data.corr())
X = data[data.columns[1:]]

y = data['exited']
X.head()
y.head()
X = X.values

y = y.values
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(10,)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, batch_size=32, epochs=15, verbose=2, validation_split=0.2)