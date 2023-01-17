import matplotlib.pyplot as plt 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import optimizers, losses, datasets

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

import numpy as np

tf.__version__
(train_images, train_labels),(test_images, test_labels) = datasets.mnist.load_data()
plt.imshow(train_images[0], cmap = "Greys") 

print("label'i =", train_labels[0])
train_images.shape
len(train_labels)
test_images.shape
len(test_labels)
shape_size = 28*28
train_images = train_images.reshape(-1, shape_size)/255.0

test_images = test_images.reshape(-1, shape_size)/255.0
train_images.shape
model = Sequential([

    Dense(512, activation='relu', input_dim = shape_size),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
model = Sequential([

    Dense(256, activation='relu', input_dim = shape_size),

    Dropout(0.2),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
from sklearn.metrics import confusion_matrix
len(test_labels)
max_deger = []

for i in range(0,len(test_labels)):

    max_deger.append(np.argmax(predictions[i]))
type(max_deger)
cm = confusion_matrix(test_labels,max_deger)

print(cm)
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img 
def img_yukle(filename):

    img = load_img(filename,color_mode = "grayscale",target_size=(28, 28))

    plt.imshow(img,cmap='Greys')

    img = img_to_array(img)

    img = img.reshape(1, 784)

    img = img.astype('float32')

    img = img / 255.0

    return img
tahmin_img=img_yukle("../input/benim-rakamlarm/3.JPG")

pred = model.predict(tahmin_img)

print(" Tahmin :",pred.argmax())