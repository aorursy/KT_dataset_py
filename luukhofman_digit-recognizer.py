import os

import numpy as np 

import matplotlib.pyplot as plt

import tensorflow as tf

import pandas as pd 

import keras

from PIL import Image

from sklearn.model_selection import train_test_split
# Importing data

data = np.genfromtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',', skip_header=True)



labels = []

pictures = np.zeros((42000, 784))



for i in range(len(data)):

    # Get label data

    labels.append(data[i][0])  

    # Remove label from array

    pictures[i] = np.delete(data[i], [0])

    

#Reshape pictures into 28,28

labels = np.asarray(labels)

# Reshape (number images, width, height, # of channels)

pictures = pictures.reshape((42000,28,28,1))
# initaite list with images

images = [image for image in pictures[0:9]]



# Plot images

for i in range(len(images)):

    plt.subplot(330 + 1 + i)

    images[i] = images[i].reshape(28,28)

    plt.imshow(images[i])

plt.show()
# Normalising

pictures = pictures.astype('float32')

pictures /= 255



# Train, val split

x_train, x_val, y_train, y_val = train_test_split(pictures, labels, test_size=0.10, random_state=42)
# Initiale deep learning model 

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10,activation='softmax'))



model.summary()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



model.compile(optimizer ='adam',

              loss ='sparse_categorical_crossentropy',

              metrics = ['accuracy'])



# Start training

model.fit(x = x_train, y = y_train, 

          batch_size = 32, epochs = 10, 

          verbose = 1, validation_data = (x_val, y_val), 

          shuffle = True, workers = 6, 

          use_multiprocessing = True)
# Import test data

test = np.genfromtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',', skip_header=True)



# Load test data into array

x_test = np.zeros((len(test), 784))

for i in range(len(test)):

    x_test[i] = test[i]



# Pre-process test

x_test = x_test.reshape(len(test),28,28,1)

x_test = x_test.astype('float32')

x_test /= 255
# Predict

results = model.predict_classes(x_test, verbose=1)

# Get labelId

sub_labels = np.arange(1, len(results) + 1)



# Write submission

submission = {"ImageId":None, "Label":None}



submission["ImageId"] = [label for label in sub_labels]

submission["Label"] = [result for result in results]



pd.DataFrame.from_dict(data=submission).to_csv('submission.csv', header=True, index=False)