import numpy as np

import pandas as pd
data =pd.read_csv("/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv",index_col=0)

data.info()
data['class'].unique()
x = data.drop(["class"],axis=1)

y = data["class"].values

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

print("Train Set Shape: ", X_train.shape)

print("Test Set Shape: ", X_test.shape)
from tensorflow.python.keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)

y_test = to_categorical(y_test, num_classes=7)
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.callbacks import EarlyStopping

from tensorflow.python.keras.layers import Dropout



early_stop = EarlyStopping(monitor='loss', patience=2)

model = Sequential()



model.add(Dense(32, activation='relu', input_shape=(6,),kernel_initializer='random_uniform'))

model.add(Dense(64, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(128, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(256, activation='relu',kernel_initializer='random_uniform'))

model.add(Dropout(0.25))

model.add(Dense(512, activation='relu',kernel_initializer='random_uniform'))

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu',kernel_initializer='random_uniform'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(64, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(128, activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(7, activation='softmax',kernel_initializer='random_uniform'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
hist = model.fit(X_train , y_train , epochs=1100, validation_split=0.2)