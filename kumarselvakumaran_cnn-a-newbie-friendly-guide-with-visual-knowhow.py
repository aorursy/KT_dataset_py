# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D

from keras.models import Model

from keras.layers import Input,InputLayer

from keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model

import h5py
def prep_data(raw):

    img_rows, img_cols = 28, 28

    num_classes = 10

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y , img_rows, img_cols, num_classes, num_images



mnist_file = "/kaggle/input/digit-recognizer/train.csv"

mnist_data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')

x, y, img_rows, img_cols, num_classes, num_images= prep_data(mnist_data)

print('input shape : {}'.format(x.shape))

print('output shape : {}'.format(y.shape))
#initializing our model

cnn_pilot = Sequential()



#adding the first convolutional layer

cnn_pilot.add(Conv2D(filters = 16,

                     kernel_size = 5,

                     activation = 'relu',

                     input_shape = (img_rows, img_cols, 1)))
#adding the first pooling layer

cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))

#adding the second convolutional layer

cnn_pilot.add(Conv2D(filters = 32,

                     kernel_size = 5,

                     activation = 'relu'))
#adding the 2nd pooling layer

cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))
#adding the flattening layer , whose output will be used as the final NN's input

cnn_pilot.add(Flatten())
#adding the hidden layer

cnn_pilot.add(Dense(units= 64,

                    activation='relu'))

cnn_pilot.add(Dense(units = num_classes,

                    activation = 'softmax'))
cnn_pilot.compile(loss = "categorical_crossentropy",

                   optimizer = 'adam',

                   metrics = ['accuracy'])
cnn_pilot.summary()

Wsave = cnn_pilot.get_weights()

print(Wsave)
history = cnn_pilot.fit(x,

              y,

              batch_size = 100,

              epochs = 100,

              validation_split = 0.2 )

print(Wsave)
plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.title('loss')

plt.legend()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.title('accuracy')

plt.legend()

plt.show()
cnn_pilot.set_weights(Wsave)



es = EarlyStopping(monitor='val_loss', mode='min', verbose= 1 , patience=5)

mc = ModelCheckpoint("best_cnn.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)



history = cnn_pilot.fit(x,

                        y,

                        batch_size = 100,

                        epochs = 100,

                        validation_split = 0.2,

                        verbose = 1,

                        callbacks = [es,mc])
cnn_best = load_model("best_cnn.h5")

_, train_acc = cnn_best.evaluate(x, y, verbose=0)

print("Train: {}".format(train_acc))

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.title('loss')

plt.legend()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.title('accuracy')

plt.legend()

plt.show()
def image_journey(img_num):

    layer_outputs = [layer.output for layer in cnn_pilot.layers[:4]]

    activation_model = keras.Model(inputs=cnn_pilot.inputs, outputs=layer_outputs)

    activations = activation_model.predict(x[img_num].reshape(1,28,28,1))

    layer_list=['1st convolution layer','1st pooling layer','2nd convolution layer','2nd pooling layer']

    fig, ax = plt.subplots(1, 1 ,figsize = (4,4))

    plt.subplots_adjust(left=0, bottom=-0.2, right=1, top=0.9,wspace=None, hspace=0.1)

    fig.suptitle('original image', fontsize=30)

    ax.imshow(x[img_num][:,:,0], cmap='gray')

    for i in range(4):

        activation = activations[i]

        activation_index=0

        fig, ax = plt.subplots(1, 6 ,figsize = (30,3))

        fig.suptitle(layer_list[i], fontsize=50)

        plt.subplots_adjust(left=0, bottom=-0.6, right=1, top=0.6,wspace=None, hspace=0.1)

        for row in range(0,6):

            ax[row].imshow(activation[0, :, :, activation_index], cmap='gray')

            activation_index += 1

image_journey(200)
mnist_test = "/kaggle/input/digit-recognizer/test.csv"

mnist_test = np.loadtxt(mnist_test, skiprows=1, delimiter=',')

num_images = mnist_test.shape[0]

out_x = mnist_test.reshape(num_images, img_rows, img_cols, 1)

out_x = out_x / 255

results = cnn_pilot.predict(out_x)

results = np.argmax(results,axis = 1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})

submissions.to_csv("submission.csv", index=False, header=True)