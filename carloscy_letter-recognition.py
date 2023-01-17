# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from PIL import Image

import matplotlib.pyplot as plt

from keras.utils import to_categorical

import tensorflow as tf

from tensorflow import keras

from time import time

import matplotlib.pyplot as plt
#loading the data:

X = []

labels = []

alldata = '../input/notMNIST_small/notMNIST_small'

for directory in os.listdir(alldata):

    for image in os.listdir(alldata + '/' + directory):

        try:

            file_path = alldata + '/' + directory + '/' + image

            img = Image.open(file_path)

            img.load()

            img_data = np.asarray(img, dtype=np.int16)

            X.append(img_data)

            labels.append(directory) #the subdirectory is the class of the image

        except Exception as e:

            pass

print("Data loaded.")
print(X[1])
plt.imshow(X[1], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()
print(labels[1])
#Data summary (note that 2 of the training samples aren't loading for some reason):

num_examples = len(X)

print("There are", num_examples, "data samples total.")

img_width = len(X[0]) 

img_height = len(X[0][0]) 

print("The size of the samples is", img_width, "by", img_height)

print("Example unprocessed images:")

plt.imshow(X[8000], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()

print("This sample has label:", labels[8000])
int_labels = [ord(c)-ord('A') for c in labels]

#now convert labels to one hot vectors: 

one_hot_labels = to_categorical(int_labels, 10)

print("The new label sample format is", one_hot_labels[1])

one_hot_labels = np.asarray(one_hot_labels)
from numpy.random import uniform



training_fraction = 0.85 #aiming for approx. 15915 training samples and 2809 validation samples

val_X = []

val_Y = []

train_X = []

train_Y = []



for i,sample in enumerate(X):

    rand = uniform()

    if rand < training_fraction:

        train_X.append(sample)

        train_Y.append(one_hot_labels[i])

    else:

        val_X.append(sample)

        val_Y.append(one_hot_labels[i])

        

val_X = np.asarray(val_X, dtype=np.int16)

train_X = np.asarray(train_X, dtype=np.int16)

val_Y = np.asarray(val_Y, dtype=np.int16)

train_Y = np.asarray(train_Y, dtype=np.int16)
train_X.shape, train_Y.shape
# Letter I

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_X[i], cmap="gray", interpolation="nearest")

    plt.xlabel(train_Y[i])

plt.show()
first_model = keras.Sequential()
train_X.shape, val_X.shape 
from tensorflow.keras.layers import Dense, Flatten, Activation 
# We add the layers by calling model.add()

#images 28x28

first_model.add(Flatten(input_shape=(28, 28)))
first_model.add(Dense(64, activation='sigmoid'))
first_model.add(Dense(10, ))
first_model.get_config()
first_model.compile(optimizer = keras.optimizers.SGD() ,

                    loss = keras.losses.categorical_crossentropy,

                    metrics = ['accuracy'])



first_model.summary()
print("Training dataset X shape:", train_X.shape)

print("Training dataset labels shape:", train_Y.shape)
model_letters = first_model.fit(x = train_X,

                                y = train_Y,

                                batch_size = 128,

                                epochs = 5,

                                validation_split = 0.2,

                                shuffle=True)
model_letters.history
print(model_letters.history.keys())

# summarize history for accuracy

plt.plot(model_letters.history['acc'])

plt.plot(model_letters.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(model_letters.history['loss'])

plt.plot(model_letters.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_loss, test_acc = first_model.evaluate(val_X, val_Y)



print('\nTest accuracy:', test_acc)

print('\nTest loss:', test_loss)
predictions = first_model.predict(val_X)
import seaborn as sns

label_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
i = 220

plt.imshow(val_X[i], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()
y = predictions[i]

x = label_names
sns.barplot(x=x, y=y)
sig_model =  keras.Sequential()



sig_model.add(Flatten(input_shape=(28, 28)))

sig_model.add(Dense(32, activation='sigmoid'))

sig_model.add(Dense(10, activation='softmax'))

sig_model.summary()
tan_model =  keras.Sequential()



tan_model.add(Flatten(input_shape=(28, 28)))

tan_model.add(Dense(32, activation='tanh'))

tan_model.add(Dense(10, activation='softmax'))

tan_model.summary()
relu_model =  keras.Sequential()



relu_model.add(Flatten(input_shape=(28, 28)))

relu_model.add(Dense(32, activation='relu'))

relu_model.add(Dense(10, activation='softmax'))

relu_model.summary()
#Compiling

sig_model.compile(optimizer= keras.optimizers.SGD(), 

              loss= keras.losses.categorical_crossentropy,

              metrics=['accuracy'])



tan_model.compile(optimizer= keras.optimizers.SGD(), 

              loss= keras.losses.categorical_crossentropy,

              metrics=['accuracy'])



relu_model.compile(optimizer= keras.optimizers.SGD(), 

              loss= keras.losses.categorical_crossentropy,

              metrics=['accuracy'])
#Training every model

print('Training Sigmoid model')

sig_history = sig_model.fit(train_X, train_Y, 

                            validation_data= [val_X, val_Y],

                            epochs=5)



print('\n Training TanH model')

tan_history = tan_model.fit(train_X, train_Y, 

                            validation_data= [val_X, val_Y],

                            epochs=5)



print('\n Training RelU model')

relu_history = relu_model.fit(train_X, train_Y, 

                            validation_data= [val_X, val_Y],

                              epochs=5)
activation_hist = [sig_history, tan_history, relu_history]

activation_labels = ['Sigmoid','Tanh','ReLU']
def compare_models(histories, model_labels):

    legend = []

    for i, hist in enumerate(histories):

        label = model_labels[i]



        plt.plot(hist.history['acc'])

        plt.plot(hist.history['val_acc'])

        plt.title('Model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        legend.append(label + ' train')

        legend.append(label + ' test')

    plt.legend(legend, loc='upper left')



    plt.show()

    for i, hist in enumerate(activation_hist):

        # summarize history for loss

        plt.plot(hist.history['loss'])

        plt.plot(hist.history['val_loss'])

        plt.title('Model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        legend.append(label + ' train')

        legend.append(label + ' test')



    plt.legend(legend, loc='upper left')

    plt.show()
compare_models(activation_hist, activation_labels)
from tensorflow.keras import layers
num_train = len(train_X)

num_val = len(val_X)



train_X = train_X.reshape((num_train, img_width, img_height, 1))

val_X = val_X.reshape((num_val, img_width, img_height, 1))

#reshape is there to include 1D channel for monochrome image
#normalize the X inputs by dividing by 255 (max pixel value) to minimize the variation bewteen

#training samples

train_X = train_X.astype('float32')

val_X = val_X.astype('float32')

train_X /= 255

val_X /= 255



print("Train X: ", train_X.shape)

print("Train Y: ", train_Y.shape)

print("Test X:", val_X.shape)

print("Test Y:", val_Y.shape)
cnn_model = keras.Sequential()
cnn_model.add(layers.Conv2D(64, (3, 3), activation= 'relu', padding = 'same', input_shape=(28, 28, 1)))

cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.summary()
cnn_model.add(Flatten())

cnn_model.add(Dense(64, activation='sigmoid'))

cnn_model.add(Dense(10, activation = 'softmax'))
cnn_model.get_config()
cnn_model.compile(optimizer = keras.optimizers.SGD(),

                    loss = keras.losses.categorical_crossentropy,

                    metrics = ['accuracy'])



cnn_model.summary()
cnn_model_history = cnn_model.fit(x = train_X,

                                y = train_Y,

                                batch_size = 64,

                                epochs = 5,

                                validation_split = 0.2,

                                shuffle=True)
cnn_model_history.history
print(cnn_model_history.history.keys())

# summarize history for accuracy

plt.plot(cnn_model_history.history['acc'])

plt.plot(cnn_model_history.history['val_acc'])

plt.plot(sig_history.history['acc'])

plt.plot(sig_history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['CNN train', 'CNN validation', 'ANN train', 'ANN validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_model_history.history['loss'])

plt.plot(cnn_model_history.history['val_loss'])

plt.plot(sig_history.history['loss'])

plt.plot(sig_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['CNN train', 'CNN validation', 'ANN train', 'ANN validation'], loc='upper left')

plt.show()
test_loss, test_acc = cnn_model.evaluate(val_X, val_Y)



print('\nTest accuracy:', test_acc)

print('\nTest loss:', test_loss)
predictions = cnn_model.predict(val_X)
X_test_plot = val_X.reshape((num_val, img_width, img_height))
i = 400

plt.imshow(X_test_plot[i], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()
y = predictions[i]

x = label_names
y
y = predictions[i]

x = label_names

sns.barplot(x=x, y=y)