#importing the required libraries

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt
#importing the dataset

data = keras.datasets.fashion_mnist
#spliting our data into test and train

(train_images, train_labels), (test_images, test_labels) = data.load_data()
np.unique(train_labels)
#this is actual label for those 10 different clothing(Given by tensorflow website itself)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images[0]
plt.imshow(train_images[7], cmap=plt.cm.binary)

plt.show()
#here we simplifying our image data so it ranges from 0 to 1 rather than 0 to 255.

train_images = train_images/255

test_images = test_images/255
#model

#In the model we are flattening our images so that the image will become a line of 784 pixels.

#we have a hidden layer of 128 neurons

#finally we use softmax to determine our clothing by probability

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(10, activation="softmax")

])



#optimizing our model by reducing loss

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



#I tried a number of epochs and 8 seems to work preety good

model.fit(train_images, train_labels, epochs=8)
test_loss, test_acc = model.evaluate(test_images, test_labels)



print(test_acc * 100)
prediction = model.predict(test_images)



for i in range(5):

    plt.grid(False)

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel("Actual: "+class_names[test_labels[i]])

    plt.title("Predicted: "+class_names[np.argmax(prediction[i])])

    plt.show()