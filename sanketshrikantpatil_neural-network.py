import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist 



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()

plt.imshow(train_images[3])

plt.colorbar()

plt.show()
train_images = train_images / 255.0



test_images = test_images / 255.0


model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),  

    keras.layers.Dense(128, activation='relu'),  

    keras.layers.Dense(10, activation='softmax') 

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=2)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 



print('Test accuracy:', test_acc)
predictions = model.predict(test_images)
def show_prediction(img):

    i=img

    plt.figure()

    plt.imshow(test_images[i])

    plt.show()

    print("the prediction is",class_names[np.argmax(predictions[i])]+" and the picture is of ",class_names[test_labels[i]]) 
show_prediction(4546)