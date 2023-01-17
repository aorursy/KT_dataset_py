import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.pipeline import Pipeline

from keras.models import Sequential

from keras.layers import LeakyReLU, Dense, Conv1D, MaxPooling1D, Dropout, Permute, Flatten, LSTM

plt.style.use('ggplot')
df_train  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

y_train = df_train['SalePrice']

df_train = df_train.drop(columns=['SalePrice'])

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

display(df_train.head(10))
one_hot_encoded_training_predictors = pd.get_dummies(df_train)

one_hot_encoded_test_predictors = pd.get_dummies(df_test)

df_train, df_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)

df_train = df_train.fillna(df_train.mean())

df_test = df_test.fillna(df_test.mean())
scaler = StandardScaler()

ids = df_test['Id']

df_train = scaler.fit_transform(df_train)

df_test = scaler.transform(df_test)

y_train = np.log(y_train)
time_steps = 1

num_features = df_train.shape[-1]

df_train = np.reshape(df_train,(-1, time_steps, num_features))

df_test = np.reshape(df_test,(-1, time_steps, num_features))
model = Sequential()



model.add(Permute((2, 1), input_shape=(time_steps, num_features)))

model.add(Conv1D(32, 2))

model.add(MaxPooling1D(2))

model.add(Conv1D(64, 2))

model.add(MaxPooling1D(2))

model.add(LSTM(32, return_sequences=False))

model.add(Dropout(0.1))

model.add(Dense(1))

model.add(LeakyReLU())

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(df_train, y_train, epochs=1000, batch_size=64, verbose=2)
predictions = pd.Series(np.exp(model.predict(df_test)).reshape((-1)), dtype='float64')

submission = pd.DataFrame({"Id": ids, "SalePrice": predictions})

submission.to_csv("submission.csv", index=False)