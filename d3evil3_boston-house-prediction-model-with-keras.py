import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('../input/boston-house-prices/housing.csv',header=None, delimiter=r"\s+",names=column_names)
df.head()
X = df.drop('MEDV',axis=1).values

y = df['MEDV'].values
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)
from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(30,activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test), epochs=600)
loss_data = pd.DataFrame(model.history.history)
loss_data.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error
predict= model.predict(X_test)
mean_absolute_error(y_test,predict)
np.sqrt(mean_squared_error(y_test,predict))
plt.figure(figsize=(12,6))

plt.scatter(y_test,predict)

plt.plot(y_test,y_test,'r')
single_house = df.drop('MEDV',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,13))
model.predict(single_house)