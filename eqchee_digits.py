import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping





%matplotlib inline

filepath = "../input/digit-recognizer"
data = pd.read_csv(os.path.join(filepath, "train.csv"))

print(data.head())
print(data.shape)
sns.countplot(data.label)
print(data.isnull().any().describe())
train, val = train_test_split(data, random_state=0)

print(train.shape)
print(val.shape)
X_train = train.drop('label', axis=1)

X_val = val.drop('label', axis=1)

y_train = train.label

y_val = val.label
X_train = X_train/255.0

X_val = X_val/255.0
X_train = X_train.values.reshape(-1,28,28,1)

print(X_train.shape)
X_val = X_val.values.reshape(-1,28,28,1)

print(X_val.shape)
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
X_train = np.asarray(X_train)

X_val = np.asarray(X_val)

y_train = np.asarray(y_train)

y_val = np.asarray(y_val)
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=1)])
pd.DataFrame(history.history).plot()

plt.show()
test = pd.read_csv(os.path.join(filepath,"test.csv"))
test = test/255.0
test = test.values.reshape(-1, 28, 28, 1)
prediction = model.predict_classes(test)
submission = pd.DataFrame({

    'ImageId': pd.Series(range(1,28001)),

    'Label': prediction

})

submission.to_csv('submission.csv', index=False)
model.save("model.h5")