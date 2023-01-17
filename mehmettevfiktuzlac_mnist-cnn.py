import matplotlib.pyplot as plt 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import optimizers, losses, datasets

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,Flatten

import numpy as np

tf.__version__
(train_images, train_labels),(test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32')

test_images = test_images.astype('float32')
train_images = train_images / 255.0

test_images = test_images / 255.0
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape = (28,28,1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])
model.fit(train_images,train_labels, epochs=10)
predictions = model.predict(test_images)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
from sklearn.metrics import confusion_matrix
max_deger = []

for i in range(0,len(test_labels)):

    max_deger.append(np.argmax(predictions[i]))
cm = confusion_matrix(test_labels,max_deger)

print(cm)
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img 
def load_image(filename):

    img = load_img(filename,color_mode = "grayscale",target_size=(28, 28))

    plt.imshow(img,cmap='Greys')

    img = img_to_array(img)

    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')

    img = img / 255.0

    return img
quary_img=load_image("../input/benim-rakamlarm/5.JPG")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())