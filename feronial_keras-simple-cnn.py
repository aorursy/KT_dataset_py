# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from PIL import Image

import cv2

import numpy as np # linear algebra

import os

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Input, Dense, Flatten

from keras.optimizers import Adadelta

from keras.losses import binary_crossentropy

from keras.layers.convolutional import Conv2D

import matplotlib.pyplot as plt

from keras.layers.pooling import MaxPooling2D

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=[]

labels=[]

parasitized_path =os.listdir("../input/cell_images/cell_images/Parasitized/")

for pars in parasitized_path:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+ pars)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((64, 64))

        #image_from_array = image_from_array.convert('L')

        data.append(np.array(size_image))

        labels.append(0)

    except Exception as e:

        print(e)

uninfected_path=os.listdir("../input/cell_images/cell_images/Uninfected/")

for unef in uninfected_path:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/" + unef)

        image_from_array = Image.fromarray(image, 'RGB')

        #image_from_array = image_from_array.convert('L')

        size_image = image_from_array.resize((64, 64))

        data.append(np.array(size_image))

        labels.append(1)

    except Exception as e:

        print(e)
print('Lenght of Data : ' + str(len(data)))

print('Lenght of Data : ' + str(len(labels)))

numpy_data = np.array(data)

numpy_labels = np.array(labels)

s = np.arange(numpy_data.shape[0])

print(s)

np.random.shuffle(s)

print(s)

numpy_data = numpy_data[s]

numpy_labels = numpy_labels[s]

data_length = len(numpy_data)
print('Lenght of Data : ' + str(len(numpy_data)))

print('Lenght of Data : ' + str(len(numpy_labels)))
training_X = numpy_data[:round(data_length * 0.95)]

training_Y = numpy_labels[:round(data_length * 0.95)]

#val_X = numpy_data[len(training_X) :round((len(training_X)) + (len(training_X) * 0.05)) ]

#val_Y = numpy_labels[len(training_Y) :round((len(training_Y)) + (len(training_Y) * 0.05))]

test_X = numpy_data[round((len(training_X))) : ]

test_Y = numpy_labels[round((len(training_Y))): ]
training_X = training_X.astype('float64') / 255

#val_X = val_X.astype('float64') / 255

test_X = test_X.astype('float64') / 255
print('Lenght of Data : ' + str(len(training_X)))

#print('Lenght of Data : ' + str(len(val_X)))

print('Lenght of Data : ' + str(len(test_X)))
visible = Input(shape=(64,64,3))



conv1 = Conv2D(64, kernel_size=4, activation='relu')(visible)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(32, kernel_size=4, activation='relu')(pool1)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(16, kernel_size=4, activation='relu')(pool2)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)





flat = Flatten()(pool3)

hidden1 = Dense(8, activation='relu')(flat)

output = Dense(1, activation='sigmoid')(hidden1)

model = Model(inputs=visible, outputs=output)

print(model.summary())
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
graph_viz = model.fit(

        training_X,

        training_Y,

        epochs = 10,

        batch_size = 64,

        validation_data=(test_X,test_Y)    

        )
plt.plot(graph_viz.history['acc'])

plt.plot(graph_viz.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(graph_viz.history['loss'])

plt.plot(graph_viz.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
import time

a = time.time()

print(model.predict(test_X[10].reshape((1,64,64,3))))

b = time.time()

c = b - a

print(c)
test_Y[10]