import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense
df=pd.read_csv('../input/Iris.csv')
df.head(3)
X = df["SepalLengthCm"]

Y = df["SepalWidthCm"]



x = X[0]

y = Y[0]



x_data = [x]

y_data = [y]

print(x_data, y_data)
model = Sequential()



l = Dense(1, activation='linear', input_dim=1)

model.add(l)



model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])



model.fit(x_data, y_data, epochs = 200)



answer = model.predict(x_data)

print('Predicted:', answer)