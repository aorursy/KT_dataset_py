import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('../input/daily_total_female_births_in_cal.csv')
df.columns = ['Births']
df_copy = df.copy()
df.head()
sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(df['Births'])
df['Births(t-0)'] = df.Births.shift(-1)
df.head()
df['Births(t-1)'] = df.Births.shift(-2)
df.head()
df['Births(t-2)'] = df.Births.shift(-3)
df['Births(t-3)'] = df.Births.shift(-4)
df.head()
df = df.dropna()
test_set = df[-30:]
train_set = df[:-30]
X_test_set = test_set.drop(['Births(t-3)'], axis=1)
y_test_set = test_set['Births(t-3)']
X_train_set = train_set.drop(['Births(t-3)'], axis=1)
y_train_set = train_set['Births(t-3)']
import warnings
warnings.filterwarnings('ignore')
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
X = array(X_train_set)
y = array(y_train_set)
X = X.reshape((X.shape[0], X.shape[1],1))
model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, validation_split=0.10)
x_test = array(X_test_set)
x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_input, verbose=0)
print(prediction)
y_test_set = array(y_test_set)
y_test_set
prediction.flatten()
sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(y_test_set, label = "Actual")
plt.plot(prediction, label = "Predicted")
plt.show()
actual = np.append(y,y_test_set)
predicted = np.append(y, prediction)
sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(actual, label = "Actual")
plt.plot(predicted, label = "Predicted", alpha=0.7)
plt.legend()
plt.show()