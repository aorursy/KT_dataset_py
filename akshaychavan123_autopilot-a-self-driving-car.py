import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from math import pi, ceil

import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import TruncatedNormal, Constant
from keras import regularizers
from keras import optimizers
import cv2
# Image list (input)
steering_x = []

# Steering angle (output)
steering_y = []

# Defining the path to dataset 
image_dir = '/kaggle/input/test-set-self-driving-cars/driving_dataset/'
# Reading the text file which contain image name and its corresponding steering angle

with open("/kaggle/input/test-set-self-driving-cars/driving_dataset/data.txt") as file:
    for lines in file:
        X, y = lines.split()
        steering_x.append(image_dir + X) # appending the image_dir along with its name to list
        steering_y.append((float(y) * pi) / 180) # appending the sttering angle after conversion to output list
# Printing the length of input and the output list
print(len(steering_x))
print(len(steering_y))
# printing few input and output from the list 

print("***** Image names ***** \n", steering_x[:10])
print("***** Steering angle ***** \n", steering_y[:10])
steering_y[500:1500]
# First 80% as train data
train_steering_x = steering_x[:int(len(steering_x) * 0.8)]
train_steering_y = steering_y[:int(len(steering_y) * 0.8)]

# Remaining 20% as test data
val_steering_x = steering_x[-int(len(steering_x) * 0.2):]
val_steering_y = steering_y[-int(len(steering_y) * 0.2):]
plt.hist(train_steering_y, bins=40, density=1, color='green', histtype ='step')
plt.hist(val_steering_y, bins=40, density=1, color='blue', histtype ='step')

plt.show()
train_mean_steering_y = np.mean(train_steering_y)
print('Test_Mean_Squared_Error(MEAN):%f' % np.mean(np.square(val_steering_y-train_mean_steering_y)) )
print('Test_Mean_Squared_Error(ZERO):%f' % np.mean(np.square(np.array(val_steering_y)-0.0)) )
print("Number of train images =", len(train_steering_x), "\nNumber of validation images =", len(val_steering_x))
# dimensions of our images.
img_width, img_height = 66, 200
input_shape = (img_width, img_height, 3)
train_data_dir = "/kaggle/input/test-set-self-driving-cars/driving_dataset/"
nb_train_samples = 36324
nb_validation_samples = 9081
epochs = 12
batch_size = 128
def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)
# Model
model = Sequential()

# Conv Layer 1
model.add(Conv2D(24, (5, 5), strides=2, input_shape=input_shape,
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
          bias_initializer=Constant(value=0.1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Conv2D(36, (5, 5), strides=2, input_shape=input_shape,
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
          bias_initializer=Constant(value=0.1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 3
model.add(Conv2D(48, (5, 5), strides=2, input_shape=input_shape,
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
          bias_initializer=Constant(value=0.1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Conv2D(64, (3, 3), input_shape=input_shape,
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
          bias_initializer=Constant(value=0.1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Conv2D(64, (3, 3), input_shape=input_shape,
          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
          bias_initializer=Constant(value=0.1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully Conected Layer 1
model.add(Dense(1164,
         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
         bias_initializer=Constant(value=0.1),
         kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Fully Conected Layer 2
model.add(Dense(100,
         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
         bias_initializer=Constant(value=0.1),
         kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Fully Conected Layer 3
model.add(Dense(50,
         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
         bias_initializer=Constant(value=0.1),
         kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1,
         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
         bias_initializer=Constant(value=0.1),
         kernel_regularizer=regularizers.l2(0.001)))
model.add(Lambda(atan_layer))
model.summary()
adam = optimizers.Adam(lr = 0.0001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse'])
def generate_arrays_from_file():
    gen_state = 0
    while 1:
        if gen_state + 128 > len(train_steering_x):
            gen_state = 0
        paths = train_steering_x[gen_state : gen_state + 128]
        y = train_steering_y[gen_state : gen_state + 128]
        X =  [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66) )) / 255.0 for x in paths]
        gen_state = gen_state + 128
        yield np.array(X), np.array(y)
def get_validation_dataset():
    images= [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66))) / 255.0 for x in val_steering_x]
    return np.array(images), np.array(val_steering_y)
train_generator = generate_arrays_from_file()
X, y = get_validation_dataset()
X.shape
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
checkpointer = ModelCheckpoint(filepath = 'model.h5',
                               monitor='val_loss', 
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False, 
                               mode='auto', 
                               period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.000001)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=(X,y),
    callbacks=[checkpointer,reduce_lr]
)
model.save_weights("weights.h5")
history_dict = history.history
history_dict.keys()
mse = history.history['mse']
val_mse = history.history['val_mse']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize = (8, 8))
epochs = range(1, len(mse) + 1)
plt.plot(epochs, mse, 'bo', label='Training MSE')
plt.plot(epochs, val_mse, 'b', label='Validation MSE')
plt.title('Training and validation MSE')
plt.legend()
plt.figure(figsize = (8, 8))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
