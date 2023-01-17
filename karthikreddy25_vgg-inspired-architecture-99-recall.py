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



# dataset download api command

# kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
pneumonia_train = list(paths.list_images("../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/"))

normal_train = list(paths.list_images("../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/"))

print(len(pneumonia_train))

print(len(normal_train))
# define classes

columns = 3

classes = {

    "Pneumonia":[cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in random.sample(pneumonia_train, columns)], 

    "Normal": [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in random.sample(normal_train, columns)]

}



# this method displays images for two classes above and below

def display(classes, columns, read_as_rgb=True, cmap=None):

    for _class in classes:

        #print(random_images)

        fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(14, 10), squeeze=False)

        fig.tight_layout()

        for l in range(1):

            for m, img in enumerate(classes[_class]):

                axes[l][m].imshow(img, cmap=cmap)

                axes[l][m].axis("off")

                axes[l][m].set_title(_class)

    # done displaying

    

# display images

display(classes, columns)
def preprocess(image, input_mode="grayscale", reshape=True):

    # convert to uint8 watch out after new definition

    img = image.astype(np.uint8)

    # if the image is of bgr then equalize each channel and join back

    if input_mode == "bgr":

        B, G, R = cv2.split(img)

        B = cv2.equalizeHist(B)

        G = cv2.equalizeHist(G)

        R = cv2.equalizeHist(R)

        img = cv2.merge([B, G, R])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # if grayscale image

    elif input_mode == "grayscale":

        img = cv2.equalizeHist(img)

        if reshape:

            img = img.reshape(224, 224, 1)

    img = img/255.

    return img
# get some random positive images

sample_positive = random.sample(pneumonia_train, 3)



# make a dictionary regular samples and preprocessed ones

classes = {

    "Original": [cv2.imread(img, 0) for img in sample_positive],

    "Preprocessed": [preprocess(cv2.imread(img, 0), input_mode="grayscale", reshape=False) for img in sample_positive]

} 



# display the images 

display(classes, 3, cmap="gray")
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img



IM_WIDTH = 224

IM_HEIGHT = 224

BATCH_SIZE = 32



try:

    from imutils import paths

except ModuleNotFoundError:

    !pip install imutils

    from imutils import paths



_generator = ImageDataGenerator(

    width_shift_range=0.01,

    height_shift_range=0.01,

    zoom_range=0.01,

    horizontal_flip=False,

    rotation_range=2.99,

)
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

np.random.seed(255)

import os

os.environ["PYTHONHASHSEED"] = str(255)

import random

random.seed(255)

import tensorflow as tf

tf.set_random_seed(255)



from keras import backend as K

tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)

K.set_session(sess)



COLOR_MODE = "grayscale"



# generator for train and validation

_image_generator = ImageDataGenerator(

    # rescale=1./255,

    width_shift_range=0.01,

    height_shift_range=0.01,

    zoom_range=0.01,

    horizontal_flip=False,

    rotation_range=2.99,

    preprocessing_function=preprocess

)



# train

_train = _image_generator.flow_from_directory(

    directory="../input/chest-xray-pneumonia/chest_xray/chest_xray/train/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=True,

    seed=255,

    color_mode=COLOR_MODE,

    class_mode="categorical",

    batch_size=BATCH_SIZE

)



# we don't need augumentation for validation and test, so we define a new generator

_test_generator = ImageDataGenerator(

    preprocessing_function=preprocess

)

# validation

_valid = _test_generator.flow_from_directory(

    directory="../input/chest-xray-pneumonia/chest_xray/chest_xray/val/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=True,

    seed=255,

    color_mode=COLOR_MODE,

    class_mode="categorical",

    batch_size=16

)
from keras import callbacks

from keras.layers import LeakyReLU

from keras.models import Sequential, Model

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
class CustomNet(object):



    def __init__(self, height, width, channels, classes, parameter_scaling):

        self.height = height

        self.width = width

        self.channels = channels

        self.output_classes = classes

        self.scale = parameter_scaling

    

    def model(self):

        input_shape = (self.height, self.width, self.channels)

        chan_dim = -1

        scale = self.scale

        

        # model architecture

        _model = Sequential()

    

        # 224x224

        # convolution 1

        _model.add(Conv2D(scale, (3, 3), input_shape=input_shape))

        _model.add(LeakyReLU(alpha=0.1))

        # convolution 2

        _model.add(Conv2D(2*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(MaxPooling2D(pool_size=(2, 2)))

        _model.add(Dropout(0.1))

        

        # 112x112

        # convolution 3

        _model.add(Conv2D(3*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        # convolution 4

        _model.add(Conv2D(4*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(MaxPooling2D(pool_size=(2, 2)))

        _model.add(Dropout(0.1))

        

        # 56x56

        # convolution 5

        _model.add(Conv2D(5*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        # convolution 6

        _model.add(Conv2D(3*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(MaxPooling2D(pool_size=(2, 2)))

        _model.add(Dropout(0.1))

        

        # 28x28

        # convolution 7

        _model.add(Conv2D(6*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(MaxPooling2D(pool_size=(2, 2)))

        _model.add(Dropout(0.1))

        

        # 14x14

        # convolution 8

        _model.add(Conv2D(7*scale, (3, 3)))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(MaxPooling2D(pool_size=(2, 2)))

        _model.add(Dropout(0.1))



        # flattening layer

        _model.add(Flatten())



        # first dense layer

        _model.add(Dense(units=15*scale))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(Dropout(0.5))



        # second dense layer

        _model.add(Dense(units=15*scale))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(Dropout(0.5))



        # third dense layer

        _model.add(Dense(units=15*scale))

        _model.add(LeakyReLU(alpha=0.1))

        _model.add(Dropout(0.5))



        # output layer

        _model.add(Dense(self.output_classes, activation="softmax"))



        # print(model.summary()) 

        return _model
LR = 1e-4

SCALE = 32

BATCH_SIZE = 32

EPOCHS = 30



def lr_scheduler(epoch, lr):

    if 20 < epoch <= 40:

        return (1e-4)*0.25

    elif 40 < epoch <= 50:

        return 1e-5

    else:

        return lr

    

# define callbacks

_callbacks = [

    callbacks.TensorBoard(

        log_dir="tensorboard", write_graph=True, write_images=False    

    ),

    LearningRateScheduler(lr_scheduler)

]



# initiate model

model = CustomNet(

    height=IM_HEIGHT, width=IM_WIDTH, channels=1, 

    classes=2, parameter_scaling=SCALE

).model()



# compile

model.compile(

    loss="binary_crossentropy",optimizer=Adam(lr=LR), metrics=["accuracy"]

)

print(_train.class_indices)
%%time

# start training

history = model.fit_generator(

    generator=_train,

    steps_per_epoch=_train.samples//BATCH_SIZE,

    validation_data=_valid,

    validation_steps=_valid.samples,

    callbacks=_callbacks,

    class_weight={

        0:3.5,

        1:1.0

    },

    epochs=EPOCHS

)
# Let's define test data from the test generator that we used for validation also

# validation

_test = _test_generator.flow_from_directory(

    directory="../input/chest-xray-pneumonia/chest_xray/chest_xray/test/",

    target_size=(IM_WIDTH, IM_HEIGHT),

    shuffle=False,

    seed=255,

    color_mode=COLOR_MODE,

    class_mode="categorical",

    batch_size=1

)

print(_test.class_indices)
test_loss, test_acc = model.evaluate_generator(_test, steps=_test.samples, verbose=1)



print('val_loss:', test_loss)

print('val_cat_acc:', test_acc)
# PREDICT

predictions = model.predict_generator(_test, steps=_test.samples, verbose=1)

print(predictions.shape)

y_pred = np.argmax(predictions, axis=1)

y_test = _test.classes
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns

from sklearn.metrics import confusion_matrix



report = classification_report(y_test, y_pred, target_names=['NORMAL', 'PNEUMONIA'])

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

print("F1 score of the model: {:.2f}".format(f1_score(y_test, y_pred)))



# precision

print("Precision score of the model: {:.2f}".format(precision_score(y_test, y_pred)))



# recall

print("Recall score of the model: {:.2f}".format(recall_score(y_test, y_pred)))