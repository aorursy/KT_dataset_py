import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM



df = pd.read_csv("../input/Merged.csv")

df.head()
df.shape
df.fillna(0, inplace=True)

print("checking if any null values are present\n", df.isna().sum())
list(df.columns.values)
df.index = df.Date

df.drop('Date', axis=1, inplace=True)

df.head()
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_df = scaler.fit_transform(df)
scaled_df[0,0]
X_train = []

y_train = []

for i in range(60, scaled_df.shape[0]):

    X_train.append(scaled_df[i-60:i, 0])

    y_train.append(scaled_df[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))