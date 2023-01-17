import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X = np.array(train.drop(["label"], axis=1)) / 255.
y = np.array(train["label"])

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

X_test = np.array(test) / 255.

print(X.shape, y.shape, X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Reshape, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
inp = Input(shape=(X_train.shape[1], ))
reshape = Reshape(target_shape=(28, 28, 1))(inp)
conv_1 = Conv2D(128, (3,3), padding="same", activation="relu")(reshape)
conv_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_1)
conv_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_2)
conv_3 = BatchNormalization()(conv_3)
pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
flatten = Flatten()(pool_3)
fc = Dense(128, activation="relu")(flatten)
outp = Dense(y_train.shape[1], activation="softmax")(fc)
model_nodrop = Model(inp, outp)

model_nodrop.summary()

model_nodrop.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('nodrop.h5', 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only = True)

hist = model_nodrop.fit(X_train, 
                        y_train, 
                        verbose=0,
                        batch_size=512, 
                        epochs=50, 
                        validation_data=(X_val, y_val), 
                        callbacks=[checkpoint])
%matplotlib inline

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure()
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()
max(val_acc)
inp = Input(shape=(X_train.shape[1], ))
reshape = Reshape(target_shape=(28, 28, 1))(inp)
conv_1 = Conv2D(128, (3,3), padding="same", activation="relu")(reshape)
conv_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_1)
conv_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_2)
conv_3 = BatchNormalization()(conv_3)
pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
flatten = Flatten()(pool_3)
fc = Dense(128, activation="relu")(flatten)
fc = Dropout(0.5)(fc)
outp = Dense(y_train.shape[1], activation="softmax")(fc)

model_drop = Model(inp, outp)

model_drop.summary()

model_drop.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
checkpoint = ModelCheckpoint('dropout.h5', 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only = True)

hist = model_drop.fit(X_train, 
                      y_train,
                      verbose=0,
                      batch_size=64, 
                      epochs=50, 
                      validation_data=(X_val, y_val),
                      callbacks=[checkpoint])
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure()
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()
max(val_acc)
# https://github.com/andry9454/KerasDropconnect/blob/master/ddrop/layers.py

from tensorflow.keras.layers import Wrapper
import tensorflow.keras.backend as K

class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)
inp = Input(shape=(X_train.shape[1], ))
reshape = Reshape(target_shape=(28, 28, 1))(inp)
conv_1 = Conv2D(128, (3,3), padding="same", activation="relu")(reshape)
conv_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_1)
conv_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_2)
conv_3 = BatchNormalization()(conv_3)
pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
flatten = Flatten()(pool_3)
fc = DropConnect(Dense(128, activation="relu"), prob=0.5)(flatten)
outp = Dense(y_train.shape[1], activation="softmax")(fc)

model_dropconn = Model(inp, outp)

model_dropconn.summary()

model_dropconn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
checkpoint = ModelCheckpoint('dropconn.h5', 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only = True)

hist = model_dropconn.fit(X_train, 
                          y_train, 
                          verbose=0,
                          batch_size=64, 
                          epochs=50, 
                          validation_data=(X_val, y_val),
                          callbacks=[checkpoint])
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure()
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()
max(val_acc)
sub_nodrop = pd.read_csv("../input/sample_submission.csv")
sub_drop = pd.read_csv("../input/sample_submission.csv")
sub_dropconn = pd.read_csv("../input/sample_submission.csv")
model_nodrop.load_weights("nodrop.h5")
model_drop.load_weights("dropout.h5")
model_dropconn.load_weights("dropconn.h5")
y_test = model_nodrop.predict(X_test, batch_size=1024, verbose=0)
sub_nodrop.Label = np.argmax(y_test, axis=1)
sub_nodrop.to_csv("submission_nodrop.csv", index=False)
y_test = model_drop.predict(X_test, batch_size=1024, verbose=0)
sub_drop.Label = np.argmax(y_test, axis=1)
sub_drop.to_csv("submission_drop.csv", index=False)
y_test = model_dropconn.predict(X_test, batch_size=1024, verbose=0)
sub_dropconn.Label = np.argmax(y_test, axis=1)
sub_dropconn.to_csv("submission_dropconn.csv", index=False)