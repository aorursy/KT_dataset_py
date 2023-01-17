import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras

%matplotlib inline
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
dress_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0

test_images = test_images / 255.0
train_images.shape
plt.imshow(train_images[10])

plt.title(train_labels[10])

plt.xlabel(dress_names[train_labels[10]])
plt.figure(figsize=(15,15))

for i in range(5):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.ylabel(train_labels[i])

    plt.xlabel(dress_names[train_labels[i]])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(64, activation=tf.nn.relu),

    keras.layers.Dense(32, activation=tf.nn.relu),

    keras.layers.Dense(16, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)



print('Test accuracy:', test_acc)
predictions = model.predict(test_images)

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

    

    plt.xlabel("{} {:2.0f}% ({})".format(dress_names[predicted_label],

                                100*np.max(predictions_array),

                                dress_names[true_label]),

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
i = 12

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(i, predictions,  test_labels)

plt.show()