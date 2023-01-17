import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os
from keras.layers import Input, Activation, add, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.core import Reshape
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

# Using a GPU for this kernel
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from keras import backend as K
K.set_image_dim_ordering('tf')
# Load up the data
train_data = np.genfromtxt('../input/train.csv', delimiter=',')[1:]
train_X = train_data[:, 1:]
train_y_orig = train_data[:, :1]

# converting to one-hot encoding 
train_y = np.zeros([train_y_orig.shape[0], 10])
for ind in range(train_y_orig.shape[0]):
    train_y[ind][int(train_y_orig[ind][0])] = 1
plt.hist(train_y_orig, color='firebrick', bins=10)
plt.xticks(range(0,10))
plt.xlabel("number")
plt.ylabel("count")
plt.rcParams["patch.force_edgecolor"] = True
plt.show()
model = Sequential()

input_shape = (28, 28, 1)

model.add(Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(1,1)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))

opt = RMSprop()
model.compile(loss='categorical_crossentropy',
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()
traingen = ImageDataGenerator(
             featurewise_center=False,  
             samplewise_center=False,  
             featurewise_std_normalization=False,  
             samplewise_std_normalization=False,  
             zca_whitening=False, 
             rotation_range=10, 
             zoom_range = 0.1, 
             width_shift_range=0.1, 
             height_shift_range=0.1, 
             horizontal_flip=False,  
             vertical_flip=False)  
batch_size=2**8
epochs = 100

# reshaping data and normalize
n = train_X.shape[0]
train_X = train_X.reshape((n, 28, 28, 1)).astype('float32') / 255

# traingen.fit(train_X)
# history = model.fit_generator(traingen.flow(train_X,train_y, batch_size=batch_size),
#                               epochs=epochs,
#                               steps_per_epoch=train_X.shape[0]//batch_size,
#                              )


history = model.fit(train_X, train_y, 
                    epochs=epochs,
                    batch_size=batch_size)
plt.plot(history.history['acc'], color="dodgerblue")
plt.ylim(0.995, 1)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history['loss'], color="firebrick")
plt.ylim(0, 0.015)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
print(model.evaluate(train_X, train_y))
# Load the test data
test_data = np.genfromtxt('../input/test.csv', delimiter=',')[1:]
predictions = model.predict(test_data.reshape((test_data.shape[0], 28, 28, 1)).astype('float32') / 255)
predictions = predictions.argmax(1)
sub_data = np.zeros([predictions.shape[0], 2])
count = 0
for val in predictions:
    sub_data[count] = [count + 1, val]
    count += 1
sub_data = sub_data.astype(int)
np.savetxt(fname="submission.csv",
           X=sub_data,
           fmt='%i',
           delimiter=',',
           comments='',
           header='ImageId,Label')