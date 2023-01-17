import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()
input_ = tf.keras.layers.Input(shape=(3,))
dense1 = tf.keras.layers.Dense(309, activation='relu')(input_)
dense2 = tf.keras.layers.Dense(30, activation='relu')(dense1)
output = tf.keras.layers.Dense(1)(dense2)
model = tf.keras.Model(inputs=input_, outputs=output)
model.summary()
model.compile('adam','mae')
def function(x, y, z):
    return 7*x**2 + 12*y**2 + 19*z**2 - 12
X = np.random.rand(1000,3)
y = function(X[:,0], X[:,1], X[:,2])
X[:,0]
X[:,1]
X[:,2]
y
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
print(f'train_X: {train_X.shape}')
print(f'test_X: {test_X.shape}')
print(f'train_y: {train_y.shape}')
print(f'test_y: {test_y.shape}')
early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
MODEL_PATH = 'checkpoints/model_at_{epoch:02d}.mdl'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH)
model.fit(train_X, train_y, validation_split=0.2, batch_size=4, epochs=150, callbacks=[early_stop, model_checkpoint])
!ls checkpoints/
SAVED_MODEL_PATH = 'model.mdl'
model.save(SAVED_MODEL_PATH)
saved_model = tf.keras.models.load_model(SAVED_MODEL_PATH)
saved_model.summary()
tf.keras.utils.plot_model(saved_model)
test_y_pred = saved_model.predict(test_X)
saved_model.evaluate(test_X, test_y)