import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Input, Flatten, Dense, Dropout, Activation

import keras

from keras.models import Model, Sequential

from keras.optimizers import RMSprop

from keras.regularizers import l2

from tqdm import tqdm_notebook as tqdm 



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dim = 64

num_classes = 15

maxepoches = 30

batch_size = 50



DATASET_PATH = '../input/15-scene/15-Scene'

one_hot_lookup = np.eye(num_classes)

dataset_x = []

dataset_y = []

for category in sorted(os.listdir(DATASET_PATH)):

    for fname in os.listdir(DATASET_PATH+"/"+category):

        img = cv2.imread(DATASET_PATH+"/"+category+'/'+fname, 2)

        img = cv2.resize(img, (dim,dim))

        dataset_x.append(np.reshape(img, [dim,dim,1]))

        dataset_y.append(np.reshape(one_hot_lookup[int(category)], [num_classes]))



dataset_x = np.array(dataset_x)

dataset_y = np.array(dataset_y)



"""shuffle dataset"""

p = np.random.permutation(len(dataset_x))

dataset_x = dataset_x[p]

dataset_y = dataset_y[p]

        

test_x = dataset_x[:int(len(dataset_x)/10)]

test_y = dataset_y[:int(len(dataset_x)/10)]

train_x = dataset_x[int(len(dataset_x)/10):]

train_y = dataset_y[int(len(dataset_x)/10):]



train_y2 = keras.utils.to_categorical(train_y).astype('float32')

test_y2 = keras.utils.to_categorical(test_y).astype('float32')
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',

                 input_shape=train_x.shape[1:]))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))

model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool'))



# Block 2

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))

model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool'))

model.add(Dropout(0.5))



# Block 3

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))

model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool'))

model.add(Dropout(0.5))





# Top layers

model.add(Flatten(name='flatten'))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))





model.summary()
#model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])

model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr=0.0001,decay=1e-5), metrics=['accuracy'])





hist = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size, verbose=1, epochs=maxepoches)
score = model.evaluate(test_x, test_y)

print("Training Accuracy: %.2f%%" % (hist.history['acc'][maxepoches - 1]*100))

print("Testing Accuracy: %.2f%%" % (score[1]*100))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(['train','test'])

plt.title('loss')

plt.savefig("loss7.png",dpi=300,format="png")

plt.figure()

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.legend(['train','test'])

plt.title('accuracy')

plt.savefig("accuracy7.png",dpi=300,format="png")

model_json = model.to_json()

with open("model7.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model7.h5")

print("Saved model to disk")