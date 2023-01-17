import os



path = '/kaggle/input/mahindra-2020-challenge'

os.listdir(path)
import pandas as pd



train_data = pd.read_csv(f"{path}/fashion-mnist_train.csv")



train_data
train_labels = train_data.label.values

train_images = train_data.drop(columns=["label"]).values



train_images.shape, train_labels.shape
import random 

import matplotlib.pyplot as plt



classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

r, c = 3, 5

plt.figure(figsize=(c*3, r*3))

for row in range(r):

    for col in range(c):

        index = c*row + col

        plt.subplot(r, c, index + 1)

        ix = random.randint(0, len(train_images)-1)

        plt.imshow(train_images[ix].reshape(28,28))

        plt.axis('off')

        plt.title(classes[train_labels[ix]])

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
import tensorflow as tf

from tensorflow import keras



X_train, X_valid = train_images[:50000] / 255., train_images[50000:] / 255.

y_train, y_valid = train_labels[:50000], train_labels[50000:]



X_train.shape, X_valid.shape
model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=[28*28]),

    keras.layers.Dense(300, activation="relu"),

    keras.layers.Dense(100, activation="relu"),

    keras.layers.Dense(10, activation="softmax")

])



model.summary()
model.compile(loss="sparse_categorical_crossentropy",

              optimizer="sgd",

              metrics=["accuracy"])



history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot()

plt.grid(True)

plt.show()
model.evaluate(X_valid, y_valid)
import pandas as pd



test_data = pd.read_csv(f"{path}/fashion-mnist_test.csv")



test_data
test_images = test_data.drop(columns=["Id"]).values

test_images = test_images / 255



test_images.shape
import numpy as np



preds = model.predict(test_images)

preds = np.argmax(preds, axis=1)

preds.shape
r, c = 3, 5

plt.figure(figsize=(c*3, r*3))

for row in range(r):

    for col in range(c):

        index = c*row + col

        plt.subplot(r, c, index + 1)

        ix = random.randint(0, len(test_images)-1)

        plt.imshow(test_images[ix].reshape(28,28))

        plt.axis('off')

        plt.title(classes[preds[ix]])

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
submission = pd.DataFrame({

    'Id': test_data["Id"].values,

    "Category": preds

})



submission
submission.to_csv("submission.csv", index=None)