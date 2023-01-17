import tensorflow as tf
tf.__version__
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model
model = Sequential([

    Flatten(input_shape = (28,28))

])
model.summary()
model = Sequential([

    Flatten(input_shape = (28,28)),

    Dense(16,activation = 'relu'),

    Dense(16,activation = 'relu'),

    Dense(10,activation = 'sigmoid')

])
model.summary()
784*16
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
model = Sequential([

    Conv2D(filters = 16,kernel_size = (3,3),input_shape =(28,28,1)),

    MaxPooling2D((3,3)),

    Flatten(),

    Dense(10,activation = 'softmax')

])
model.summary()
model = Sequential([

    Conv2D(filters = 16,kernel_size = (3,3),padding = 'SAME',input_shape =(28,28,1)),

    MaxPooling2D((3,3)),

    Flatten(),

    Dense(10,activation = 'softmax')

])
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 5e-3)

mae = tf.keras.metrics.MeanAbsoluteError()
model.compile(optimizer = opt,

             loss = 'sparse_categorical_crossentropy',

             metrics = [mae])
model.optimizer, model.loss
model.metrics
# Load the Fashion-MNIST dataset



fashion_mnist_data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()
from tensorflow.keras.preprocessing import image
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Print the shape of the training data

train_images.shape
# Define the labels



labels = [

    'T-shirt/top',

    'Trouser',

    'Pullover',

    'Dress',

    'Coat',

    'Sandal',

    'Shirt',

    'Sneaker',

    'Bag',

    'Ankle boot'

]
# Rescale the image values so that they lie in between 0 and 1.

train_images =  train_images / 255.

test_images = test_images / 255.
# Display one of the images

i =0

img = train_images[i,:,:]

plt.imshow(img)

plt.show()

print(f'Label is {labels[train_labels[i]]}')
history = model.fit(train_images[...,np.newaxis],train_labels,epochs = 10, batch_size =256)
df = pd.DataFrame(history.history)
df
df.plot(y = "loss",title = "Loss vs. Epochs",xlabel = "Epochs")
test_loss, test_mae = model.evaluate(test_images[...,np.newaxis], test_labels, batch_size=128)

print(test_loss,test_mae)
inx = 30
test_img = test_images[inx,:,:]

plt.imshow(test_img)

plt.show()

print(f'Label is {labels[test_labels[inx]]}')
predictions = model.predict(test_img[np.newaxis,...,np.newaxis])
predictions
print(f'Predicted label is {labels[np.argmax(predictions)]}')