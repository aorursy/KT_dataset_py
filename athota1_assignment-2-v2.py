from six.moves.urllib.request import urlretrieve

import os

import numpy as np

import io, gzip, requests, gc

import keras

import keras.backend as K

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Model, Sequential

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Input, Flatten, Dense, Dropout, Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import SGD

from keras.applications.vgg16 import VGG16

import cv2

import h5py as h5py

from keras.callbacks import EarlyStopping

from sklearn import manifold

from time import time

import matplotlib.pyplot as plt

from matplotlib import offsetbox

np.random.seed(962342)



train_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"

train_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"

test_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"

test_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

def extract_labels(url, num_images):

    print (url)

    filepath, _ = urlretrieve(url)

    statinfo = os.stat(filepath)

    with gzip.open(filepath) as bytestream:

        bytestream.read(8)

        buf = bytestream.read(1 * num_images)

        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        os.remove (filepath)

        return labels

def extract_data(url, num_images):

    print (url)

    filepath, _ = urlretrieve(url)

    statinfo = os.stat(filepath)

    with gzip.open(filepath) as bytestream:

        bytestream.read(8)

        buf = bytestream.read(28 * 28 * num_images)

        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        data = data.reshape(num_images, 28,28)

        os.remove (filepath)

    return data





train_labels = extract_labels(train_label_url, 60000)

train_images_raw = extract_data(train_image_url, 60000)



test_labels = extract_labels(test_label_url, 10000)

test_images_raw = extract_data(test_image_url, 10000)

train_images = train_images_raw.reshape(len(train_labels), 28*28)

test_images = test_images_raw.reshape(len(test_labels), 28*28)

X_train = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

X_test = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

X_train /= 255

X_test /= 255

X_train -= 0.5

X_test -= 0.5

X_train *= 2.

X_test *= 2.

Y_train = train_labels

Y_test = test_labels

Y_train2 = keras.utils.to_categorical(Y_train).astype('float32')

Y_test2 = keras.utils.to_categorical(Y_test).astype('float32')
plt.rcParams['figure.figsize']=(20, 10)

# Scale and visualize the embedding vectors

def plot_embedding1(X, Image, Y, title=None):

    x_min, x_max = np.min(X, 0), np.max(X, 0)

    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    ax = plt.subplot(111)

    for i in range(X.shape[0]):

        plt.text(X[i, 0], X[i, 1], str(Y[i]),

                 color=plt.cm.Set1(Y[i] / 10.),

                 fontdict={'weight': 'bold', 'size': 9})



    if hasattr(offsetbox, 'AnnotationBbox'):

        # only print thumbnails with matplotlib > 1.0

        shown_images = np.array([[1., 1.]])  # just something big

        for i in range(X.shape[0]):

            dist = np.sum((X[i] - shown_images) ** 2, 1)

            if np.min(dist) < 4e-3:

                # don't show points that are too close

                continue

            shown_images = np.r_[shown_images, [X[i]]]

            imagebox = offsetbox.AnnotationBbox(

                offsetbox.OffsetImage(Image[i], cmap=plt.cm.gray_r),

                X[i])

            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])

    if title is not None:

        plt.title(title)

num_classes = len(set(Y_train))



img_input = Input(shape = X_train.shape[1:]) 



vgg16 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)

vgg16 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(vgg16)

vgg16 = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(vgg16)



# Block 2

vgg16 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(vgg16)

vgg16 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(vgg16)

vgg16 = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool')(vgg16)



# Block 3

vgg16 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(vgg16)

vgg16 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(vgg16)

vgg16 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(vgg16)

vgg16 = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool')(vgg16)

'''

# Block 4

vgg16 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(vgg16)

vgg16 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(vgg16)

vgg16 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(vgg16)

vgg16 = MaxPooling2D((3, 3), strides=(2, 2), name='block4_pool')(vgg16)



# Block 5

vgg16 = Conv2D(512, (2, 2), activation='relu', padding='same', name='block5_conv1')(vgg16)

vgg16 = Conv2D(512, (2, 2), activation='relu', padding='same', name='block5_conv2')(vgg16)

vgg16 = Conv2D(512, (2, 2), activation='relu', padding='valid', name='block5_conv3')(vgg16)

vgg16 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(vgg16)

'''

# Top layers

vgg16 = Flatten(name='flatten')(vgg16)

vgg16 = Dense(1024, activation='relu')(vgg16)

vgg16 = Dropout(0.5)(vgg16)

vgg16 = Dense(1024, activation='relu')(vgg16)

vgg16 = Dropout(0.5)(vgg16)



vgg16 = Dense(num_classes, activation='softmax')(vgg16)
model = Model(img_input, vgg16)



model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])

model.summary()



modelfit = model.fit(X_train, Y_train2, validation_data=(X_test, Y_test2), batch_size=100, verbose=1, epochs=20)



score = model.evaluate(X_test, Y_test2)

print(score)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import os

from six.moves.urllib.request import urlretrieve

from keras import optimizers



from sklearn import svm

from sklearn.metrics import classification_report,accuracy_score



from keras.utils.data_utils import get_file

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import VGG16



from six.moves.urllib.request import urlretrieve

import os

import numpy as np

import io, gzip, requests, gc

import keras

import keras.backend as K

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Model, Sequential

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Input, Flatten, Dense, Dropout, Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import SGD

from keras.applications.vgg16 import VGG16

import cv2

import h5py as h5py

import keras.models

from keras.callbacks import EarlyStopping

import keras

from matplotlib import pyplot as plt

import numpy as np

import gzip

%matplotlib inline

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D

from keras.models import Model

from keras.optimizers import RMSprop

from keras.applications.vgg16 import preprocess_input

import tensorflow as tf

import numpy as np

import cv2

from keras import optimizers

from sklearn.model_selection import train_test_split



# Create dictionary of target classes

label_dict = {

 0: 'A',

 1: 'B',

 2: 'C',

 3: 'D',

 4: 'E',

 5: 'F',

 6: 'G',

 7: 'H',

 8: 'I',

 9: 'J',

}



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

   

BASE_URL = 'http://yann.lecun.com/exdb/mnist/'



def increse_size(images, num_images):

    train_data = []

    for img in images:

        resized_img = cv2.resize(img, (224, 224))

        train_data.append(resized_img)

    train_data = np.vstack(train_data)

    return train_data  

def extract_data(file_name, num_images):

    dimention_size = 28

    url = BASE_URL + file_name

    print (url)

    filepath, _ = urlretrieve(url)

    statinfo = os.stat(filepath)

    with gzip.open(filepath) as bytestream:

        bytestream.read(16)

        buf = bytestream.read(dimention_size * dimention_size * num_images)

        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        data = data.reshape(num_images, dimention_size, dimention_size)

        os.remove (filepath)

    return increse_size(data, num_images)

def extract_labels(file_name, num_images):

    url = BASE_URL + file_name

    print (url)

    filepath, _ = urlretrieve(url)

    statinfo = os.stat(filepath)

    with gzip.open(filepath) as bytestream:

        bytestream.read(8)

        buf = bytestream.read(1 * num_images)

        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        os.remove (filepath)

        return (labels, num_images)

train_data = extract_data('train-images-idx3-ubyte.gz', 6000)

print ('Training images loaded' )

test_data = extract_data('t10k-images-idx3-ubyte.gz', 999)

print ('Testing images loaded')



train_labels = extract_labels('train-labels-idx1-ubyte.gz',6000)

print ('Training labels loaded')

test_labels = extract_labels('t10k-labels-idx1-ubyte.gz',999)

print ('Testing labels loaded')



# Shapes of training set

print("Training set (images) shape: {shape}".format(shape=train_data.shape))

# Shapes of test set

print("Test set (images) shape: {shape}".format(shape=test_data.shape))
train_data = train_data.reshape(-1, 224,224, 1)

test_data = test_data.reshape(-1, 224,224, 1)

train_data.shape, test_data.shape
plt.figure(figsize=[5,5])



# Display the first image in training data

plt.subplot(121)

curr_img = np.reshape(train_data[0], (224,224))

#curr_img = train_data[0]

curr_lbl = train_labels[0]

plt.imshow(curr_img, cmap='gray')

#plt.title("(Label1: " + str(label_dict[curr_lbl]) + ")")

#plt.title("(Label2: " + str(curr_lbl) + ")")



# Display the first image in testing data

plt.subplot(122)

curr_img = np.reshape(test_data[0], (224,224))

#curr_img = test_data[0]

curr_lbl = test_labels[0]

plt.imshow(curr_img, cmap='gray')

#plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

#plt.title("(Label: " + str(curr_lbl) + ")")



batch_size = 128

epochs = 30

inChannel = 1

x, y = 224, 224

input_img = Input(shape = (x, y, inChannel))



train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,

                                                             train_data, 

                                                             test_size=0.2, 

                                                             random_state=13)



vgg16 = VGG16(include_top=False,input_shape = (224, 224, 3))

vgg16.summary()



model = Model(input_img, vgg16.layers[0](input_img))



model.compile(optimizer=optimizers.RMSprop(lr=2e-4),

              loss='mean_squared_error',

              metrics=['acc'])

autoencoder_train = model.fit(train_X, train_X, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_X))

loss = autoencoder_train.history['loss']

val_loss = autoencoder_train.history['val_loss']

plt.figure()

plt.plot(range(len(loss)), loss, 'bo', label='Training loss')

plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import VGG16

# load the model

model = VGG16()

# load an image from file

image = load_img('../input/tiger.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array

image = img_to_array(image)

# reshape data for the model

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model

image = preprocess_input(image)

# predict the probability across all output classes

yhat = model.predict(image)

# convert the probabilities to class labels

label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability

label = label[0][0]

# print the classification

print('%s (%.2f%%)' % (label[1], label[2]*100))