# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import h5py
from sklearn.model_selection import train_test_split
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import tensorflow as tf

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")
train1 = train.copy()
Y_train = train1["label"]
X_train = train1.drop(labels = ["label"],axis = 1) 
X_train = X_train/255
X_train = X_train.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
Y_train
random_seed = 2
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
def digit_recog_model(input_shape,classes = 10):
    X_input = Input(input_shape)
    X = X_input
    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1,1), name = '1',padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '2')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1,1), name = '3', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = '5',padding = 'valid')(X)
    X = Dropout(0.25)(X)
    X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), name = '6', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '7')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), name = '8', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '9')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = '10',padding = 'valid')(X)
    X = Dropout(0.25)(X)
    """X = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), name = '11', padding = 'same', kernel_initializer =tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '12')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), name = '13', padding = 'same', kernel_initializer =tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '14')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), name = '15', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '16')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = '17',padding = 'valid')
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '18', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '19')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '20', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '21')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '22', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '23')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = '24',padding = 'valid')
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '25', padding = 'same', kernel_initializer =tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '26')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '27', padding = 'same', kernel_initializer =tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '28')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), name = '29', padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())(X)
    X = BatchNormalization(axis = 3, name = '30')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = '31',padding = 'valid')"""
    X = Flatten()(X)
    X = Dense(256, activation='relu', name = '33')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name = '34')(X)
    model = Model(inputs = X_input, outputs = X,name = 'digit_vgg16')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
"""checkpoint_path = "../input/digit_checkpoints_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)"""
model = digit_recog_model(input_shape = (28,28,1), classes = 10)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86), epochs = 35,steps_per_epoch=X_train.shape[0]//86, callbacks=[learning_rate_reduction])
test = test / 255.0
test = test.values.reshape(-1,28,28,1)
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_augmented_v4.csv",index=False)

