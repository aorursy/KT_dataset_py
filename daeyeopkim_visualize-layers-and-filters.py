import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as k
from tensorflow import keras

print(tf.__version__)

dataset_mnist = keras.datasets.fashion_mnist
(train_images, train_labels) , (test_images, test_label) = dataset_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))

for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(np.squeeze(train_images[i]), cmap = plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])

train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images / 255.0
test_images = test_images.reshape(test_images.shape[0],28,28,1)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, 'relu'))
model.add(keras.layers.Dropout(rate = 0.2))
model.add(keras.layers.Dense(10, 'softmax'))

model.compile(optimizer='Adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics= ['accuracy'])
model.summary()
train_images.shape, train_labels.shape , type(train_images), type(train_labels)
model.fit(train_images, train_labels , batch_size=100, epochs=5)
model.evaluate(test_images, test_label, verbose = 2)
def layer_to_visualize(layer, image_number):
  image = train_images[image_number]
  to_visualize = k.function([model.input], [layer.output])
  visuals = to_visualize([np.expand_dims(image, axis=0)])[0]

  filter_size = visuals.shape[-1]
  n = np.ceil(np.sqrt(filter_size))
  
  plt.figure(figsize=(13,13))
  for i in range(filter_size):
    plt.subplot(n,n,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(visuals[0, :, :, i], cmap = 'gray')
  
layer_to_visualize(model.layers[1], 9)
l1 = model.layers[0]
w1, b1 = l1.get_weights()

plt.figure(figsize=(13,13))
n = np.ceil(np.sqrt(w1.shape[-1]))
print(w1.shape)

for i in range(w1.shape[-1]):
  plt.subplot(n,n,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(w1[:,:,0,i], cmap = plt.cm.binary)