import keras.backend as K

import math

import numpy as np

import h5py

import matplotlib.pyplot as plt



import numpy as np

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline



def mean_pred(y_true, y_pred):

    return K.mean(y_pred)



def load_dataset():

    train_dataset = h5py.File('../input/happy-house-dataset/train_happy.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('../input/happy-house-dataset/test_happy.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()



# Normalize image vectors

X_train = X_train_orig/255.

X_test = X_test_orig/255.



# Reshape

Y_train = Y_train_orig.T

Y_test = Y_test_orig.T



print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
print("Image shape :",X_train_orig[525].shape)

imshow(X_train_orig[525])

print(Y_train[525])
def HappyModel(input_shape):

    """

    Implementation of the HappyModel.

    

    Arguments:

    input_shape -- shape of the images of the dataset

        (height, width, channels) as a tuple.  

        Note that this does not include the 'batch' as a dimension.

        If you have a batch like 'X_train', 

        then you can provide the input_shape using

        X_train.shape[1:]

    """



    #Returns:

    #model -- a Model() instance in Keras

    

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!

    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((3, 3))(X_input)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool')(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='HappyModel')



    # Feel free to use the suggested outline in the text above to get started, and run through the whole

    # exercise (including the later portions of this notebook) once. The come back also try out other

    # network architectures as well. 



    return model
happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
happyModel.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 50)
preds = happyModel.evaluate(x = X_test, y = Y_test)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
print("Image shape :",X_train_orig[525].shape)

#imshow(X_test_orig[525])

print("Actual ",Y_test[99])



img = X_test_orig[99]

imshow(img)



x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



print("Predicted ",happyModel.predict(x))