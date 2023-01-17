import numpy as np

import pandas as pd

import tensorflow as tf



from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.utils.np_utils import to_categorical





import keras.backend as K

K.set_image_data_format('channels_last')



from matplotlib.pyplot import imshow

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



# settings

EPOCH = 20    

BATCH_SIZE = 128

VALIDATION_SIZE = 2000 # set to 0 to train on all available data



# read training data from CSV file 

data = pd.read_csv('../input/train.csv')



print(data.shape)

data.head()
images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



print(images.shape)
# in this case all images are square

# get the width and height of the image

image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
images_r = images.reshape(images.shape[0], image_height, image_width, 1)

print(images_r.shape)
#labels_flat = data[[0]].values.ravel()

labels_flat = data.iloc[:,0].values.ravel()



labels= to_categorical(labels_flat)

labels.shape

# split data into training & validation

#print(images_r.shape)

validation_images = images_r[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = images_r[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]



train_images.shape
# define the Keras model

def train_model(input_shape):

    """

    Implementation of the train_model.

    

    Arguments:

    input_shape -- shape of the images of the dataset



    Returns:

    model -- a Model() instance in Keras

    """

     

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!

    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((1, 1))(X_input)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool')(X)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(64, (11, 11), strides = (1, 1), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    

    # DROPOUT

    X = Dropout(0.25)(X)

    

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(128, activation='relu', name='fc1')(X)

     

    # DROPOUT

    X = Dropout(0.5)(X)

        

    X = Dense(10, activation='softmax', name='fc2')(X)



    # Create model. 

    model = Model(inputs = X_input, outputs = X, name='NMIST')



    return model

    
NN_model = train_model((28,28,1))

NN_model.compile(optimizer="adam", loss="categorical_crossentropy",metrics = ["accuracy"])
NN_model.fit(x=train_images,y=train_labels,epochs=EPOCH,batch_size=BATCH_SIZE)
NN_model.summary()
NN_model.evaluate(x=validation_images, y =validation_labels)
# read test data from CSV file 

test_images = pd.read_csv('../input/test.csv').values

test_images = test_images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

test_images = np.multiply(test_images, 1.0 / 255.0)

test_images_r =  test_images.reshape(test_images.shape[0], image_height, image_width, 1)





print(test_images_r.shape)



predicted_labels = NN_model.predict(test_images_r)



predicted_labels_ind = np.argmax(predicted_labels,axis=-1)



# save results

np.savetxt('submission_softmax.csv', 

           np.c_[range(1,len(test_images)+1),predicted_labels_ind], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')