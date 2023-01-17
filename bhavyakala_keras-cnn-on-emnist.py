#loading Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

from sklearn.model_selection import train_test_split

import os
#Preparing Data

train_data = pd.read_csv("/kaggle/input/emnist/emnist-balanced-train.csv",delimiter = ',')

test_data = pd.read_csv("/kaggle/input/emnist/emnist-balanced-test.csv", delimiter = ',')

mapp = pd.read_csv("/kaggle/input/emnist/emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)





X_train = train_data.iloc[:,1:].values

y_train = train_data.iloc[:,0].values

X_test = test_data.iloc[:,1:].values

y_test = test_data.iloc[:,0].values

H = 28

W = 28



# The images are originally rotated so this converts them to proper orientation

def rotate(image):

    image = image.reshape([H,W])

    image = np.fliplr(image)

    image = np.rot90(image)

    return image



X_train = np.apply_along_axis(rotate,1,X_train)

X_test = np.apply_along_axis(rotate,1,X_test)



# Plot example images

for i in range(10,19) :

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i])

    plt.title(chr(mapp[y_train[i]]))



num_classes = len(np.unique(y_train))



# Reshaping for input to CNN

X_train = X_train.reshape([-1,H,W,1])

X_test = X_test.reshape([-1,H,W,1])



#Rescaling

X_train = X_train.astype('float32')

X_train /=255.0

X_test = X_test.astype('float32')

X_test /=255.0



# One hot Encoding

y_train = tf.keras.utils.to_categorical(y_train,num_classes)

y_test = tf.keras.utils.to_categorical(y_test,num_classes)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size=0.1)





print("X_train: ", X_train.shape)

print("y_train: ", y_train.shape)

print("X_valid: ", X_valid.shape)

print("y_valid: ", y_valid.shape)

print("X_test: ", X_test.shape)

print("y_test: ", y_test.shape)

# Model architecture

def model_arch(input_shape) :

    input_x = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.Conv2D(32,(5,5),padding='same',activation='relu')(input_x)

    X = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(X)

    X = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')(X)

    X = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(X)

    X = tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu')(X)

    X = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(256,activation='relu')(X)

    X = tf.keras.layers.Dense(128,activation='relu')(X)

    X = tf.keras.layers.Dense(num_classes,activation='softmax')(X)

    model = tf.keras.Model(inputs=input_x, outputs=X)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.summary()

    return model

model = model_arch((H,W,1))
# Training the model

callbacks = [

    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 4,

                          verbose = 1,

                          mode = 'auto'),

    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 3)

]

history = model.fit( X_train,y_train,

           batch_size=64,

           epochs=10, 

           callbacks=callbacks,

           validation_data=(X_valid,y_valid))

# Plot accuracy, validation accuracy



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# Plot loss, validation loss



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Evaluate model on Test set

scores = model.evaluate(X_test,y_test)
# Print scores

print(scores)

print("accuracy :{:.3f}".format(scores[1]*100))