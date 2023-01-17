from __future__ import absolute_import, division, print_function, unicode_literals

#Install Tensorflow

!pip install grpcio --upgrade

!pip install google-auth --upgrade

!pip install tensorflow==2.1



import tensorflow as tf
#recognising handwritten data from MNIST Dataset

mnist = tf.keras.datasets.mnist



(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_test=x_train/255.0, x_test/255.0
#using Keras to build the Sequential Model

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)), #vector of size 784 

    tf.keras.layers.Dense(128, activation='relu'), #the dense layer 128 units: Hidden Layer

    tf.keras.layers.Dropout(0.2), #Dropout Regularization

    tf.keras.layers.Dense(10, activation='softmax') #10 units: 1 corresponding to each digit 0 to 9

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#Training of the model

model.fit(x_train, y_train, epochs=5)

#Evaluation of Model

model.evaluate(x_test, y_test)

#Accuracy= 97.72%
#For Individual Testing

from matplotlib import pyplot as plt

import numpy as np
first_image = x_test[0]

first_image = np.array(first_image, dtype='float')

pixels = first_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()
y_test[0]

#predicted correctly
x_test.shape
yp=model.predict(x_test)
yp[0]

#here, the highest probability is of a 7