import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
car_df = pd.read_csv('../input/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

car_df[:10]
sns.pairplot(car_df)
X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
X.head()
y = car_df['Car Purchase Amount']

y.shape
from sklearn.preprocessing import MinMaxScaler



scaler_x = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)

scaler_x.data_max_
scaler_x.data_min_
print(X_scaled)
X_scaled.shape
y.shape
y = y.values.reshape(-1,1)
y.shape
#first ten value

y[:10]
scaler_y = MinMaxScaler()



y_scaled = scaler_y.fit_transform(y)

y_scaled[:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)
import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler



model = Sequential()

model.add(Dense(128, input_dim=5, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=24,  verbose=1, validation_split=0.2)

print(epochs_hist.history.keys())

plt.plot(epochs_hist.history['loss'])

plt.plot(epochs_hist.history['val_loss'])



plt.title('Model Loss Progression During Training/Validation')

plt.ylabel('Training and Validation Losses')

plt.xlabel('Epoch Number')

plt.legend(['Training Loss', 'Validation Loss'])

# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 



# ***(Note that input data must be normalized)***



X_test_sample = np.array([[0, 0.4370344,  0.53515116, 0.57836085, 0.22342985]])

#X_test_sample = np.array([[1, 0.53462305, 0.51713347, 0.46690159, 0.45198622]])



y_predict_sample = model.predict(X_test_sample)



print('Expected Purchase Amount=', y_predict_sample)

y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)

print('Expected Purchase Amount=', y_predict_sample_orig)
End