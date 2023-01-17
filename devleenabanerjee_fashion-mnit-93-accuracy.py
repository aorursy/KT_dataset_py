import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Dense,Flatten,Dropout
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pandas as pd
train_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

train_images=train_data.drop('label',axis=1)
test_images=test_data.drop('label',axis=1)
train_label = train_data['label']
test_label = test_data['label']
train_images = np.array(train_images).astype('float32')
test_images = np.array(test_images).astype('float32')
train_images = train_images.reshape(60000, 28, 28)
test_images = test_images.reshape(10000, 28, 28)
train_images, test_images = train_images / 255.0, test_images / 255.0
#Get the initial 25 pics from the datast
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_label[i]])

plt.show()
#Get the shape of the train dataset
train_images.shape
#Get the shape of the test dataset
test_images.shape
	# reshape dataset to have a single channel

train_images=train_images.reshape(len(train_images),28,28,1)
test_images=test_images.reshape(len(test_images),28,28,1)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
model.add(Dropout(0.5))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
epochs = 20  # for better result increase the epochs
batch_size=100
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])
model.fit(
  train_images,
  to_categorical(train_label),
  epochs=epochs,
  batch_size=batch_size,
  validation_data=(test_images, to_categorical(test_label)),
)
#make Prediction
# predict the class
result = model.predict(test_images)