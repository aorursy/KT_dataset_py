!pip install tensorflow==2.2.0
import tensorflow as tf

import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import itertools



print(tf.__version__)
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()

y_test = y_test.flatten()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize=(10,7))

p = sns.countplot(y_train.flatten())

p.set(xticklabels=classes)
np.isnan(x_train).any()
np.isnan(x_test).any()
input_shape = (32, 32, 3)



x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)

x_train=x_train / 255.0

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

x_test=x_test / 255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)

y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
y_train[0]
plt.imshow(x_train[100])

print(y_train[100])
batch_size = 32

num_classes = 10

epochs = 50
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax'),

])



model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06),

            loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size,

                    epochs=epochs)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")

legend = ax[1].legend(loc='best', shadow=True)
test_loss, test_acc = model.evaluate(x_test, y_test)
# Predict the values from the validation dataset

y_pred = model.predict(x_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1)

# compute the confusion matrix

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 9))

c = sns.heatmap(confusion_mtx, annot=True, fmt='g')

c.set(xticklabels=classes, yticklabels=classes)