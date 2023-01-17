# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
df = pd.read_csv('../input/TWHM-WebAttacksTrain.csv', index_col='IDNum')
import re
def to_int(obj):
    return int(re.sub("[^\d]", '', obj))

df['Source IP'] = df['Source IP'].apply(to_int)
df['Destination IP'] = df['Destination IP'].apply(to_int)
df['Timestamp'] = df['Timestamp'].apply(to_int)

df = df.drop(['Usage'], axis=1)
df = df.drop(['Flow ID'], axis=1)

df = df.replace([np.nan, 'Infinity'], 2**31-1)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
enc = LabelEncoder()
y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
y_test_ohe = ohe.fit_transform(y_test.reshape(-1, 1))
y_ohe = ohe.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=83))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train_scale,
                    y_train_ohe,
                    epochs=80,
                    batch_size=512,
                    validation_data=(X_test_scale, y_test_ohe))
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'bo', label='Training loss')
plt.semilogy(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
enc.inverse_transform(model.predict_classes(X_scale))
dftrain = pd.DataFrame(enc.inverse_transform(model.predict_classes(X)), columns=['Label'], index=X.index).reset_index()
dftrain[dftrain['Label'] != 'BENIGN']
datatest = pd.read_csv('../input/TWHM-WebAttacksTestPrivat.csv', index_col='IDNum')
datatest['Source IP'] = datatest['Source IP'].apply(to_int)
datatest['Destination IP'] = datatest['Destination IP'].apply(to_int)
datatest['Timestamp'] = datatest['Timestamp'].apply(to_int)

datatest = datatest.drop(['Usage'], axis=1)
datatest = datatest.drop(['Flow ID'], axis=1)

datatest = datatest.replace([np.nan, 'Infinity'], 2**31-1)
datatest_scale = scaler.transform(datatest)
dftest = pd.DataFrame(enc.inverse_transform(model.predict_classes(datatest_scale)), columns=['Label'], index=datatest.index).reset_index()
pd.read_csv('../input/TWHM-WebAttacksKeySample.csv')
an = pd.concat([dftest, dftrain])
an.sort_values('IDNum')
an
an.to_csv('submission.csv', index=False)
