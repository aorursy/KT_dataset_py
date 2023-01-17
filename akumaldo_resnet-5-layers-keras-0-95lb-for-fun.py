# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"

from sklearn.model_selection import train_test_split



# Common imports

import numpy as np

import os



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils, to_categorical

from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model

from keras.initializers import glorot_uniform



import scipy.misc



from matplotlib.pyplot import imshow

import keras.backend as K





K.set_image_data_format('channels_last')

K.set_learning_phase(1)



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input")) #showing all the files in the ../input directory



# Any results you write to the current directory are saved as output. Kaggle message :D
input_path = '../input/'

train_path = input_path + 'train/train/'

test_path = input_path + 'test/test/'



train_dir="../input/train/train"

test_dir="../input/test/test"

train=pd.read_csv('../input/train.csv')



sub_df=pd.read_csv('../input/sample_submission.csv')



train_id = train['id']

labels = train['has_cactus']

test_id = sub_df['id']
train.head()
x_train, x_val, y_train, y_val = train_test_split(train_id, labels, test_size=0.2)
#credits to filipmg

## function to read image and create lists of images according to the ids provided

def get_images(ids, filepath):

    arr = []

    for img_id in ids:

        img = plt.imread(filepath + img_id)

        arr.append(img)

    

    arr = np.array(arr).astype('float32')

    arr = arr / 255

    return arr
x_train = get_images(ids=x_train, filepath=train_path)

x_val = get_images(ids=x_val, filepath=train_path)

test = get_images(ids=test_id, filepath=test_path)
#identity_block



def identity_block(X, f, filters, stage, block):

    """

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    # Second component of main path

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X
#convolutional_block



def convolutional_block(X, f, filters, stage, block, s = 2):

    """

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)



    # Second component of main path

    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    ##### SHORTCUT PATH ####

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X
def ResNet50(input_shape = (32, 32, 3), classes = 1):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Keras

    """

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')



    # Stage 3

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')



    # Stage 4 

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')



    # Stage 5 

    X = convolutional_block(X, f = 3, filters = [256,256, 1024], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [256,256, 1024], stage=5, block='b')

    X = identity_block(X, 3, [256,256, 1024], stage=5, block='c')



    # AVGPOOL

    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)

    



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)

    

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
model = ResNet50(input_shape = (32,32,3), classes = 1)
callbacks = [EarlyStopping(monitor='val_loss', patience=25),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),

            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=2, mode='max')]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
img_aug = ImageDataGenerator(rotation_range=20, vertical_flip=True, horizontal_flip=True)

img_aug.fit(x_train)



batch_size = 64

epochs = 30

steps = x_train.shape[0] // batch_size



history = model.fit_generator(img_aug.flow(x_train, y_train, batch_size=batch_size), 

                    steps_per_epoch=steps, epochs=epochs,  validation_steps= 50,

                    validation_data=(x_val, y_val), callbacks=callbacks, 

                    verbose=2)
fig = plt.subplots(figsize=(12,10))

plt.plot(history.history['loss'], color='b', label="Training loss")

plt.plot(history.history['val_loss'], color='r', label="validation loss")

plt.legend(loc='best', shadow=True)
fig = plt.subplots(figsize=(12,10))

plt.plot(history.history['acc'], color='b', label="Training accuracy")

plt.plot(history.history['val_acc'], color='r', label="Validation accuracy")

plt.legend(loc='best', shadow=True)
model.load_weights("best_model.h5")



submission = model.predict(test,verbose=1)
sub_binary = [0 if value<0.50 else 1 for value in submission] 
sub_df['has_cactus'] = sub_binary

sub_df.to_csv("submission.csv",index=False)
df_sub.head()