import tensorflow as tf

from tensorflow import keras



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train
train_images = np.reshape(train[train.columns[1:]].values, (-1,28,28))

train_labels = train['label'].values



test_images = np.reshape(test[test.columns[1:]].values, (-1,28,28))

test_labels = test['label'].values
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure()

plt.imshow(train_images[0], cmap=plt.cm.binary)

plt.xticks([])

plt.yticks([])

plt.xlabel(class_names[train_labels[0]])

plt.grid(False)

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
probability_model = tf.keras.Sequential([model,

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
img = test_images[0]

img = np.reshape(img, (-1, 28, 28))
predictions_single = probability_model.predict(img)

if np.argmax(predictions_single) == test_labels[0]:

    print(f"Correct prediction: {class_names[test_labels[0]]}")

else:

    print(f"Wrong prediction: {class_names[test_labels[0]]}")