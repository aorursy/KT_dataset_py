from __future__ import  absolute_import , division , print_function

import tensorflow as tf

from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt 
mnist=keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images[0].shape
len(train_labels)
#Preprocessing the data

plt.imshow(train_images[0])
train_labels[0]
plt.imshow(test_images[0])
test_labels[0]
#buildi

train_images=train_images/255.0

test_images=test_images/255.0
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(train_labels[i])

plt.show()
train_images[0].shape
model=keras.Sequential([

                       keras.layers.Flatten(input_shape=(28,28)),

                       keras.layers.Dense(256,activation=tf.nn.relu),

                       keras.layers.Dense(10,activation=tf.nn.softmax),

                       ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

  
model.fit(train_images,train_labels,epochs=5)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print("Test accuracy:",test_acc)
predictions=model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
plt.imshow(test_images[0])
def plot_image(i,predictions_array,true_label,img):

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  plt.imshow(img[i],cmap=plt.cm.binary)

  predicted_label=np.argmax(predictions_array[i])

  if predicted_label==true_label[i]:

    plt.xlabel(" {}".format(predicted_label),color="blue")

  else:

    plt.xlabel(" {}".format(predicted_label),color="red")
i = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions, test_labels, test_images)

plt.show()
num_rows=2

num_cols=3

num_images=num_rows*num_cols



plt.figure(figsize=(num_rows,num_cols))

for i in range(num_images):

  plt.subplot(num_rows,num_cols,i+1)

  plot_image(i, predictions, test_labels, test_images)

  plt.show()

  