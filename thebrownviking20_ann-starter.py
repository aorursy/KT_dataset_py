# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
# Importing data
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
X_train = df.iloc[:,1:-1].values
y_train = df.iloc[:,-1].values
y_train = np_utils.to_categorical(y_train)
# Create model
model = Sequential()
model.add(Dense(1024, input_dim=54, kernel_initializer='uniform', activation='selu'))
model.add(Dense(512, kernel_initializer='uniform', activation='softplus'))
model.add(Dense(256, kernel_initializer='uniform', activation='elu'))
model.add(Dense(128, kernel_initializer='uniform', activation='selu'))
model.add(Dense(64, kernel_initializer='uniform', activation='softplus'))
model.add(Dense(32, kernel_initializer='uniform', activation='elu'))
model.add(Dense(16, kernel_initializer='uniform', activation='softplus'))
model.add(Dense(8, kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model
model.fit(X_train, y_train, epochs=500, batch_size=32)
df_test = pd.read_csv('../input/test.csv')
X_test = df_test.iloc[:,1:].values
preds = model.predict(X_test)
sub = pd.DataFrame({"Id": df_test.iloc[:,0].values,"Cover_Type": np.argmax(preds,axis=1)})
sub.to_csv("etc.csv", index=False) 