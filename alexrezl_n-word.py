from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow и tf.keras

import tensorflow as tf

from tensorflow import keras



# Вспомогательные библиотеки

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



print(tf.__version__)
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train.head()
test.head()
#fashion_mnist = keras.datasets.fashion_mnist



#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



train_labels = train.label

test_labels = test.label

train_images = train.drop(['label'], axis=1)

test_images = test.drop(['label'], axis=1)
train_labels.head(5)
train_images = train_images.to_numpy().reshape(60000, 28, 28)

test_images = test_images.to_numpy().reshape(10000, 28, 28)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255.0



test_images = test_images / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nТочность на проверочных данных:', test_acc)
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
def plot_image(i, predictions_array, true_label, img):

  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])



  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'



  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

  predictions_array, true_label = predictions_array[i], true_label[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)



  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
i = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(i, predictions,  test_labels)

plt.show()
i = 12

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(i, predictions,  test_labels)

plt.show()
# Отображаем первые X тестовых изображений, их предсказанную и настоящую метки.

# Корректные предсказания окрашиваем в синий цвет, ошибочные в красный.

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions, test_labels, test_images)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions, test_labels)

plt.show()
# Берем одну картинку из проверочного сета.

img = test_images[0]



print(img.shape)
# Добавляем изображение в пакет данных, состоящий только из одного элемента.

img = (np.expand_dims(img,0))



print(img.shape)
predictions_single = model.predict(img)



print(predictions_single)
plot_value_array(0, predictions_single, test_labels)

_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])