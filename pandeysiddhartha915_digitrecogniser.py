# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()
train = train.sample(frac=1)
train.head()
X = train.iloc[:,1:].to_numpy()

Y = train.iloc[:, 0].to_numpy()

print(Y.shape)



X = X/255.0

X = X.reshape(X.shape[0], 28, 28, 1)

print(X.shape)
X_test = test.to_numpy()



X_test = X_test/255.0

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(X_test.shape)
num_classes = 10



Y = tf.one_hot(Y, num_classes)

print(Y.shape)
split_ratio = 0.75



train_size = int(X.shape[0] * split_ratio)

print(train_size)
X_train = X[:train_size, :, :, :]

Y_train = Y[:train_size, :]

print(X_train.shape)



X_val = X[train_size:, :, :, :]

Y_val = Y[train_size:, :]

print(X_val.shape)
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, log={}):

    if(log.get("acc") == 1.0):

      print("Reached 100% accuracy so cancelling training!")

      self.model.stop_training=True
callback = myCallback()
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape = (28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
num_epochs = 30
history = model.fit(X_train, Y_train, epochs=num_epochs, validation_data=(X_val, Y_val), callbacks=[callback], verbose=1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation Loss')

plt.legend(loc=0)

plt.figure()





plt.show()
Y_test = model.predict(X_test)
predictions = np.argmax(Y_test, axis = 1)

predictions = predictions.reshape(predictions.shape[0], 1)
np.savetxt("pred.csv", predictions, delimiter=",")
test.head()