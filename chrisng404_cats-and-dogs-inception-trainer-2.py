# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# print(os.listdir("../input/images/images"))



# Any results you write to the current directory are saved as output.
# from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.models import load_model



from collections import Counter

from scipy.ndimage import imread

from scipy.io import loadmat

from scipy.misc import imresize

import os

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import numpy
base_model = Xception(weights='imagenet', include_top=False)
# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

# x = Dropout(0.5)(x)

x = Dense(100, activation='relu')(x)

# This layer being at 1024 isn't bad. ^

x = Dropout(0.5)(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(37, activation='softmax')(x)

# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



model.summary()



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["acc"])
# load images

image_dir = "../input/images/images"

basenames = [f for f in os.listdir(image_dir) if(f[-3:] == "jpg")][:]

targs = ['_'.join(f.split("_")[:-1]) for f in basenames]

image_paths = [os.path.join(image_dir, f) for f in basenames]







images = [imread(imp) for imp in image_paths]

grays = {i for i, im in enumerate(images) if(len(im.shape) != 3)}

images = [im for i, im in enumerate(images) if(i not in grays)]

targs = [t for i, t in enumerate(targs) if(i not in grays)]

images = [im if(im.shape[2]==3) else im[:,:,:-1] for im in images]

# resize all images

im_size = (200,200)

# try 224x224, cause that's what VGG uses and we're going to be swapping models

images = [imresize(im, size=im_size) for im in images]



#def data_gen():

#    while(True):

        #Choose N random images from training set

        # read those images

        # preprocess, convert cats to one-hot, etc

#        yield (images, labels)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')

temp = Counter(targs)

print(temp)
def preprocess_input(x):

    y = x.astype("float16")

    y /= 255.

    y -= 0.5

    y *= 2.

    return y



images = [preprocess_input(im) for im in images]
# Encode labels

targ_encoder = LabelBinarizer()

labels = targ_encoder.fit_transform(targs)

# split into testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.40, stratify=labels)
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='bestmodel2.hdf5', 

                               verbose=1, save_best_only=True)

#model.fit([train_vgg19, train_resnet50], train_targets, 

#          validation_data=([valid_vgg19, valid_resnet50], valid_targets),

#          epochs=10, batch_size=4, callbacks=[checkpointer], verbose=1)

model.fit(

    x=numpy.array(X_train),

    y=Y_train,

    epochs=2,

    # Best Epoch: 500?

    batch_size=8,

    #Best Batch_size: 8

    callbacks=[checkpointer], 

    verbose=1,

    validation_data=(numpy.array(X_test), Y_test)

)
# save the output

#model.save("Puppyzon.h5")
# Load the model

model = load_model("bestmodel2.hdf5")
# Quantify Model

X_test = numpy.array(X_test)

Y_test = numpy.array(Y_test)

# Evaluate the model using the training data

model_loss, model_accuracy = model.evaluate(X_test, Y_test, verbose=2)

print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
# Make Prediction
