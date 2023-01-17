import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib as pt

import matplotlib.pyplot as plt

import seaborn as sns
print("Tensorflow: " + tf.__version__)

print("-------------------------------\n")

print("Numpy: " + np.__version__)

print("-------------------------------\n")

print("Matplotlib: " + pt.__version__)

print("-------------------------------")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print("The pixcel size for the each image in Fashion MNIST is: 28 x 28 ")

print("-----------------------------------------------------------------\n")



print("Total number of the training dataset and the pixel size: " + str(len(train_images)))

print("-----------------------------------------------------------------\n")



print("Total number of the testing dataset and the pixel size: " + str(len(test_images)))

print("-----------------------------------------------------------------")
plt.figure(figsize=(15, 10))

for i in range(20):

    plt.subplot(5, 5, i + 1)

    plt.grid(False)

    plt.imshow(train_images[i])

plt.show()
plt.figure(figsize=(15, 10))

for i in range(20):

    plt.subplot(5, 5, i + 1)

    plt.grid(False)

    plt.imshow(train_images[i], cmap = plt.cm.binary)

plt.show()
train_images = train_images / 255.0



test_images = test_images / 255.0
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



model.save("Fashion-MNIST.h5")



print('\nTest accuracy:', test_acc)
prediction = model.predict([test_images])

plt.figure(figsize=(15, 19))

for i in range(30):

    plt.subplot(5, 6, i + 1)

    plt.grid(False)

    plt.imshow(test_images[i],cmap = plt.cm.binary)

    plt.title("prediction: "+ class_names[np.argmax(prediction[i])] +"\n-----------\n" +"Actual: " + class_names[test_labels[i]])

plt.show()