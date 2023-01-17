import tensorflow as tf
import pandas as pd
data = pd.read_csv('../input/Iris.csv')
data
cols = data.columns
features = cols[1:5]
labels = cols[5]
import numpy as np

from pandas import get_dummies
indices = data.index.tolist()
indices
indices = np.array(indices)
np.random.shuffle(indices)
X = data.reindex(indices)[features]
y = data.reindex(indices)[labels]

y
y = get_dummies(y)

y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)
X_train = np.array(X_train).astype(np.float32)

X_test  = np.array(X_test)

y_train = np.array(y_train).astype(np.float32)

y_test  = np.array(y_test)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
X_train[3]
from tensorflow.keras import layers
model = tf.keras.Sequential([

  layers.Dense(32, input_shape=(4,)),

  layers.Dense(128, activation='relu'),

  layers.Dense(128, activation='relu'),

  layers.Dense(3, activation='softmax')

])



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train,y_train,

          validation_data=(X_test,y_test),

          epochs=50)