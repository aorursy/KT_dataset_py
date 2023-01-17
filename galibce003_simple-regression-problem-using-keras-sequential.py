import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv('../input/random-dataset/data.csv')
df.head()
df.isnull().sum()
df.shape
X = pd.DataFrame(df.iloc[:, 0:1])
y = pd.DataFrame(df.iloc[:, 1:2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 1)
X_train.shape
model = Sequential()
model.add(Dense(400, input_dim=1, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(1, activation='linear'))
keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mean_absolute_percentage_error'])
model.summary()
history = model.fit(X_train, y_train, epochs=500, batch_size=32,validation_split=0.15,validation_data=None,verbose=1)
keras.backend.clear_session()
y_pred = model.predict(X_test)
dfff = pd.DataFrame(np.c_[y_test, y_pred], columns = ['Actual', 'Predicted'])
dfff.head()
plt.plot(dfff['Actual'], color = 'Black', label = 'Actual')
plt.plot(dfff['Predicted'], color = 'Red', label = 'Predicted', alpha = 0.6)
plt.legend()

plt.show()