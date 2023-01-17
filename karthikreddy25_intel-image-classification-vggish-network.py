%%capture

import os

import sys

import random

import numpy as np

import pandas as pd

import cv2

from keras.preprocessing.image import img_to_array

try:

    from imutils import paths

except ImportError:

    !pip install imutils

    from imutils import paths

    

import matplotlib.pyplot as plt

%matplotlib inline
# lets start by examining the number of images available in training set

dir_train = "../input/intel-image-classification/seg_train/seg_train/"

for segment in os.listdir(dir_train):

    print("'{}' images in '{}' category".format(

        len(list(paths.list_images(dir_train + segment))),

        segment

    ))

print("Total '{}' images in training set".format(len(list(paths.list_images(dir_train)))))
class_list = os.listdir(dir_train)



# number of images to display for each class

columns = 3



# input shape 150x150

shape = (150, 150)



# let's create a dictionary with all the classes and some random images to display

classes = { 

    cls : [

        # read each mage and change the default colorspace

        cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in random.sample(

            # randomly sample images (length of columns) per class

            list(paths.list_images(dir_train + cls)), columns

        )

    ] for cls in class_list

}



# method to displays images for 'len(classes)' classes 'len(columns)' images per row

def display(classes, columns, cmap=None, figsize=(10, 10)):

    for _class in classes:

        # print(random_images)

        fig, axes = plt.subplots(

            nrows=1, ncols=columns, 

            figsize=figsize, squeeze=False

        )

        fig.tight_layout()

        for l in range(1):

            for m, img in enumerate(classes[_class]):

                # set map

                if len(img.shape) == 2:

                    cmap = "gray"

                axes[l][m].imshow(img, cmap=cmap)

                axes[l][m].axis("off")

                axes[l][m].set_title(_class)

    # done displaying

    

# display images

display(classes, columns)
# let us read a random image from a class and examine it for preprocessing

key = random.choice(list(classes.items()))[0]

image = random.sample(list(classes[key]), 1)
# let's display the image dimension and the image itself

display({key:image}, 1, figsize=(4, 4))

print("Dimensions: " + str(image[0].shape))
def preprocess(image, reshape=True):

    # copy for later usage?

    img = image.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # split

    h,s,v = cv2.split(img)

    # histogram equalization on value 

    v_ = cv2.equalizeHist(v)

    # merge back

    img = cv2.merge([h,s,v_])

    # to rgb

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # apply gaussian blur

    img = cv2.GaussianBlur(img, (3, 3), 1)

    # to grayscale

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #rescale

    img = img/255.

    # done

    return img 



img = image[0]

# before after image display

display({"before": [img], "after": [preprocess(img, reshape=False)]}, 1, figsize=(4,4))
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf



# seed constant for reproducable results

SEED = 255

np.random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

tf.random.set_seed(SEED)

# bunch of other stuff

from keras import backend as K

tf_config = tf.compat.v1.ConfigProto(

    intra_op_parallelism_threads=1, 

    inter_op_parallelism_threads=1, 

    device_count = { "CPU": 1,"GPU": 1 } 

)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=tf_config)

tf.compat.v1.keras.backend.set_session(sess)



IM_WIDTH = 128

IM_HEIGHT = 128

BATCH_SIZE = 32



def preprocess_grayscale(image, reshape=True):

    # copy for later usage?

    img = image.astype(np.uint8)

    # split

    # h,s,v = cv2.split(img)

    # histogram equalization on value 

    # v_ = cv2.equalizeHist(v)

    # merge back

    # img = cv2.merge([h,s,v_])

    # to rgb

    # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # histogram equalization on value 

    # img = cv2.equalizeHist(img)

    # apply gaussian blur

    img = cv2.GaussianBlur(img, (5, 5), 1)

    if reshape:

        img = img.reshape(IM_WIDTH, IM_HEIGHT, 3)

    #rescale

    img = img/255.

    # done

    return img 



# generator for training

_generator = ImageDataGenerator(

    preprocessing_function=preprocess_grayscale,

    validation_split=0.2

)



_train = _generator.flow_from_directory(

    directory="../input/intel-image-classification/seg_train/seg_train/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=True,

    seed=SEED,

    color_mode="rgb",

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    subset="training"

    

)



_valid = _generator.flow_from_directory(

    directory="../input/intel-image-classification/seg_train/seg_train/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=True,

    seed=SEED,

    color_mode="rgb",

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    subset="validation"

)



# generator for validation and test sets

_test_generator = ImageDataGenerator(

    preprocessing_function=preprocess_grayscale

)



# test set

_test = _generator.flow_from_directory(

    directory="../input/intel-image-classification/seg_test/seg_test/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=False,

    seed=SEED,

    color_mode="rgb",

    class_mode="categorical",

    batch_size=1

)



_train.class_indices
from keras import callbacks

from keras.regularizers import l1, l2

from keras.layers import LeakyReLU, ReLU

from keras.models import Sequential, Model

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.applications import ResNet50 as resnet



# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())
class Network(object):



    def __init__(self, height, width, channels, classes, parameter_scaling):

        self.height = height

        self.width = width

        self.channels = channels

        self.output_classes = classes

        self.scale = parameter_scaling



    def model(self, weights="imagenet"):

        # initiate model

        _model = Sequential()

        input_shape = (self.height, self.width, self.channels)

        ### --

        axis = -1

        alpha = 0.1

        # reg = l2(1e-4)

        reg = None

        # convinience for stacking conv blocks

        def conv_block(model, scale, axis, input_shape=None):

            if input_shape:

                model.add(

                    Conv2D(scale, (3,3), padding="same", input_shape=input_shape, kernel_regularizer=reg)

                )

            else:

                model.add(Conv2D(scale, (3,3), padding="same", kernel_regularizer=reg))

            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=axis))

            # conv_2

            model.add(Conv2D(scale, (3, 3), padding="same", kernel_regularizer=reg))

            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=axis))

            # pool_1

            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Dropout(0.1))

            return model

        # 128x128

        _model = conv_block(_model, self.scale, axis, input_shape=input_shape)

        # ~64x64

        _model = conv_block(_model, self.scale*(2**1), axis)

        # ~32x32

        _model = conv_block(_model, self.scale*(2**2), axis)

        # ~16x16

        _model = conv_block(_model, self.scale*(2**3), axis)

        # ~8x8

        # _model = conv_block(_model, self.scale*16, axis)

        # ~4x4

        # _model = conv_block(_model, self.scale*32, axis)

        # ~2x2

        # _model = conv_block(_model, self.scale*64, axis)

        

        # Fully connected layers

        _model.add(Flatten())

        

        # FC1

        _model.add(Dense(256, kernel_regularizer=reg))

        _model.add(Activation("relu"))

        _model.add(Dropout(0.25))

        # FC2

        _model.add(Dense(256, kernel_regularizer=reg))

        _model.add(Activation("relu"))

        _model.add(Dropout(0.25))

        

        # classifier

        _model.add(Dense(self.output_classes))

        _model.add(Activation("softmax"))



        # return model

        return _model
%%time

LR = 1e-3

SCALE = 16

BATCH_SIZE = 64

EPOCHS = 35



def lr_scheduler(epoch, lr):

    if epoch > 15:

        return (1e-4)

    elif epoch > 25:

        return (1e-5)

    else:

        return lr

    

# define callbacks

_callbacks = [LearningRateScheduler(lr_scheduler)]



# initiate model

model = Network(

    height=IM_HEIGHT, width=IM_WIDTH, 

    channels=3, classes=6, 

    parameter_scaling=SCALE

).model()



# compile

model.compile(

    loss="categorical_crossentropy",optimizer=Adam(lr=LR), metrics=[categorical_accuracy]

)

# print(_train.class_indices)



# start training

with tf.device("/device:GPU:0"):

    history = model.fit_generator(

        generator=_train,

        steps_per_epoch=_train.samples//BATCH_SIZE,

        validation_data=_valid,

        validation_steps=_valid.samples//64,

        callbacks=_callbacks,

        epochs=EPOCHS,

        verbose=1

    )
# Let us plot the loss and accuracy functions for both training and validation sets to see how our model behaved over training epochs

# PLOTTING

import matplotlib.pyplot as plt

%matplotlib inline



# plot the loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()



# plot the accuracy

plt.plot(history.history['categorical_accuracy'])

plt.plot(history.history['val_categorical_accuracy'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='lower right')

plt.show()
test_loss, test_acc = model.evaluate_generator(_test, steps=_test.samples, verbose=1)



print('val_loss:', test_loss)

print('val_cat_acc:', test_acc)
# PREDICT

predictions = model.predict_generator(_test, steps=_test.samples, verbose=1)

# print(predictions.shape)

y_pred = np.argmax(predictions, axis=1)

y_test = _test.classes
# confusion matrix

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns

from sklearn.metrics import confusion_matrix



report = classification_report(y_test, y_pred, target_names=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])

print(report)
# heatmap

sns.heatmap(

    confusion_matrix(y_test, y_pred), 

    annot=True, 

    fmt="d", 

    cbar = False, 

    cmap = plt.cm.Blues

)
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



# accuracy score

print("Accuracy score of the model: {:.2f}".format(accuracy_score(y_test, y_pred)))



# f1 score

print("F1 score of the model: {:.2f}".format(f1_score(y_test, y_pred, average="weighted")))



# precision

print("Precision score of the model: {:.2f}".format(precision_score(y_test, y_pred, average="weighted")))



# recall

print("Recall score of the model: {:.2f}".format(recall_score(y_test, y_pred, average="weighted")))
