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
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
input_file = "/kaggle/input/car-price/CarPrice_Assignment.csv"
pd.options.display.max_columns = None
main_df = pd.read_csv(input_file)
main_df
main_df.drop(columns=['CarName'], inplace=True)
main_df
encoding_columns = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']  # List of categorical columns
main_array = np.array(main_df.car_ID).reshape(-1, 1)
for column in main_df.columns:
    if column in encoding_columns:
        temp = np.array(pd.get_dummies(main_df[column]))  # If column is a categorical, perform one-hot encoding
    else:
        temp = np.array(main_df[column]).reshape(-1, 1)
    main_array = np.hstack((main_array, temp))  # concantate the columns
main_array = main_array[:, 2:]  # Remove car_ID column
pd.DataFrame(main_array)  # Display array
X_data = main_array[:, :-1]
y_data = main_array[:, -1].reshape(-1, 1)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_data_scaled = x_scaler.fit_transform(X_data)
y_data_scaled = y_scaler.fit_transform(y_data)
print("Shape of X_data: {}".format(X_data.shape))
print("Shape of y_data: {}".format(y_data.shape))
print("==========X_data after rescaling===============")
print(pd.DataFrame(X_data_scaled).head())
print("==========y_data after rescaling===============")
print(y_data_scaled.ravel())
X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data_scaled, test_size=0.1, shuffle=False)
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of y_test: {}".format(y_test.shape))
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(None, 57)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
history = model.fit(x=X_train, y=y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend()
plt.show()
PREDICT_ROW = 200
predict_data = main_array[PREDICT_ROW, :].reshape(1, -1)
X_predict = predict_data[:, :-1]
y_true = predict_data[:, -1]
predict_data_scaled = x_scaler.transform(X_predict)
y_pred_scaled = model.predict(predict_data_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
print("Prediction price result: {}".format(float(y_pred)))
print("True price: {}".format(float(y_true)))
print("Percentage error: {}".format(str(float(abs(y_true - y_pred) * 100 / y_true))))