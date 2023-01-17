 !pip install imutils
import argparse

import numpy as np

import os

import matplotlib.pyplot as plt

from imutils import paths

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.optimizers import SGD

import warnings

warnings.filterwarnings('ignore')

import os

import cv2

import numpy as np





class SimpleDatasetLoader:

    # Method: Constructor

    def __init__(self, preprocessors=None):

        """

        :param preprocessors: List of image preprocessors

        """

        self.preprocessors = preprocessors



        if self.preprocessors is None:

            self.preprocessors = []



    # Method: Used to load a list of images for pre-processing

    def load(self, image_paths, verbose=-1):

        """

        :param image_paths: List of image paths

        :param verbose: Parameter for printing information to console

        :return: Tuple of data and labels

        """

        data, labels = [], []



        for i, image_path in enumerate(image_paths):

            image = cv2.imread(image_path)

            label = int(image_path.split(os.path.sep)[-2])



            if self.preprocessors is not None:

                for p in self.preprocessors:

                    image = p.preprocess(image)



            data.append(image)

            labels.append(label)



            if verbose > 0 and i > 0 and (i+1) % verbose == 0:

                print('[INFO]: Processed {}/{}'.format(i+1, len(image_paths)))



        return (np.array(data), np.array(labels))







import cv2

from tensorflow.keras.preprocessing.image import img_to_array





class ImageToArray:

    def __init__(self, data_format=None):

        self.data_format = data_format



    def preprocess(self, image):

        image = img_to_array(image,data_format=self.data_format)

        return image
import cv2



class ImageResize:

    def __init__(self,width,height,inter=cv2.INTER_AREA):

        self.width = width

        self.height = height

        self.inter = inter



    def preprocess(self,image):

        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization

from tensorflow.keras.layers import MaxPool2D, Dropout, Flatten, Dense

from tensorflow.keras.models import Sequential

from keras import backend as K





class MiniVGGNet:

    @staticmethod

    def build(width, height, depth, classes):

        input_shape = (height, width, depth)

        ch_dim = -1

        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            ch_dim = 1

        model = Sequential()

        model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=ch_dim))



        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=ch_dim))



        model.add(MaxPool2D((2,2)))

        model.add(Dropout(0.25))



        model.add(Conv2D(32, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=ch_dim))



        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=ch_dim))



        model.add(MaxPool2D((2, 2)))

        model.add(Dropout(0.25))



        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=ch_dim))

        model.add(Dropout(0.5))

        model.add(Dense(classes))

        model.add(Activation('softmax'))



        return model











datasetpath = '../input/flowers17/17flowers/jpg'

model_output = 'kaggle/working/model'
imagepaths = list(paths.list_images(datasetpath))



ir = ImageResize(32,32)

iat = ImageToArray()

lo = SimpleDatasetLoader([ir,iat])

(data,labels)=lo.load(imagepaths,verbose=100)

image = cv2.imread(imagepaths[0])

imgplot = plt.imshow(image)

plt.show()
train_X,test_X,train_y,test_y = train_test_split(data,labels,test_size=0.2,random_state=123)



train_X = train_X.astype("float") / 255.0

test_X = test_X.astype("float") / 255.0
print(train_X.shape)



print(train_y.shape)



print(test_X.shape)

print(test_y.shape)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=17)



model.compile(loss='sparse_categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])


model.summary()
H = model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=70,batch_size=32)


plt.figure(figsize=(10,8))

plt.plot(np.arange(0,70),H.history['accuracy'],label='Training accuracy')



plt.plot(np.arange(0,70),H.history['loss'],label='Training loss')





plt.plot(np.arange(0,70),H.history['val_accuracy'],label='validation accuracy')





plt.plot(np.arange(0,70),H.history['val_loss'],label='Training accuracy')



plt.xlabel('#Epochs')

plt.ylabel('percentage')

plt.legend()

plt.show()