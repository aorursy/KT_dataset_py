# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt
test_set_raw = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

train_set_full_raw = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

train_set_full_raw.head()
X_train_full = np.array(train_set_full_raw.filter(like="pixel")).reshape((-1,28,28,1))/255.

y_train_full = np.array(train_set_full_raw["label"])



X_train, X_val = X_train_full[:50000], X_train_full[50000:]

y_train, y_val = y_train_full[:50000], y_train_full[50000:]



X_test = np.array(test_set_raw.filter(like="pixel")).reshape((-1,28,28,1))/255.

y_test = np.array(test_set_raw["label"])



X_train.shape, y_train.shape
batch_size = 32



train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))

train_set = train_set.shuffle(len(X_train)).batch(batch_size).prefetch(1)



val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(1)

test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(1)
plt.figure(figsize=(7,7))



for X,y in train_set.take(1):

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(X[i][:,:,0], cmap="binary")

        plt.title("{}".format(y[i]))

        plt.axis("off")
model = keras.models.Sequential([

                                 keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),

                                 keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),

                                 keras.layers.MaxPooling2D(),

                                 keras.layers.Flatten(),

                                 keras.layers.Dropout(0.25),

                                 keras.layers.Dense(128, kernel_initializer="he_normal"),

                                 keras.layers.BatchNormalization(),

                                 keras.layers.Activation("relu"),

                                 keras.layers.Dropout(0.5),

                                 keras.layers.Dense(10, activation="softmax")

])



#Tensorboard callback

# run_index = 1

# run_logdir = os.path.join(os.curdir, "fashion_mnist_logs", "run_{:03d}".format(run_index))

# tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir, histogram_freq=1)



early_stopping_cb = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

checkpoint_cb = keras.callbacks.ModelCheckpoint("fashion_mnist_model.h5", save_best_only=True)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)





callbacks = [early_stopping_cb, checkpoint_cb, lr_scheduler]



model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

model.fit(train_set, epochs=50, validation_data=val_set, callbacks=callbacks)



test_loss, test_acc = model.evaluate(test_set, verbose=2)

print('\nTest accuracy:', test_acc)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



y_pred = []

for i in model.predict(test_set.take(1)):

    y_pred.append(class_names[np.argmax(i)])





plt.figure(figsize=(10,10))



for X,y in test_set.take(1):

    for i in range(15):

        plt.subplot(3,5, i+1)

        plt.imshow(X[i][:,:,0], cmap="binary")

        plt.title("pred: {}\ntrue: {}".format(y_pred[i], class_names[y[i]]))

        plt.axis("off")
from sklearn.metrics import confusion_matrix



y_pred_2 = []

for i in model.predict(test_set):

    y_pred_2.append(np.argmax(i))







con_mat = confusion_matrix(y_test, y_pred_2)

plt.figure(figsize=(10,10))

plt.imshow(con_mat, cmap="hot" )

plt.yticks([0,1,2,3,4,5,6,7,8,9],labels=class_names)

plt.xticks([0,1,2,3,4,5,6,7,8,9],labels=class_names)

plt.ylabel("True label")

plt.xlabel("Predicted label")

plt.colorbar()