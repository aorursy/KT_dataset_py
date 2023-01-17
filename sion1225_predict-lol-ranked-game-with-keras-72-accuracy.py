from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()
print(data.shape)
Victory=data['blueWins'] #Victory=label
gameId=data['gameId']
data.drop(['gameId'],1,inplace=True)
data.drop(['blueWins'],1,inplace=True)
print(data.shape)
print(Victory.shape)
print(Victory)
mean=data.mean(axis=0)
std=data.std(axis=0)
N_data=(data-mean)/std

N_data.head()
model = Sequential()
model.add(Dense(8, input_shape=(38,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(N_data, Victory, batch_size=100, epochs=4, validation_split=0.2)#, callbacks=[early_stop])
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()