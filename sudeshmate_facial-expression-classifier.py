# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("[INFO] loading input data...")

f = open("/kaggle/input/facial-expression/fer2013.csv")

f.__next__() # f.next() for Python 2.7

(trainImages, trainLabels) = ([], [])

(valImages, valLabels) = ([], [])

(testImages, testLabels) = ([], [])



for row in f:

    (label,image,usage) = row.strip().split(",")

    label=int(label)

    image = np.array(image.split(" "), dtype="uint8")

    image = image.reshape((48, 48))

    

    if usage == 'Training':

        trainImages.append(image)

        trainLabels.append(label)

    elif usage == 'PrivateTest':

        valImages.append(image)

        valLabels.append(label)

    else:

        testImages.append(image)

        testLabels.append(label)

datasets = [

(trainImages, trainLabels, 'train'),

(valImages, valLabels, 'validation'),

(testImages, testLabels, 'test')]
%%writefile hdf5datasetwriter.py

import h5py

import os



class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images",

    bufSize=500):

        # check to see if the output path exists, and if so, raise

        # an exception

        if os.path.exists(outputPath):

            raise ValueError("The supplied ‘outputPath‘ already "

            "exists and cannot be overwritten.Manually delete "

            "the file before continuing.", outputPath)

        self.db = h5py.File(outputPath, "w")

        self.data = self.db.create_dataset(dataKey, dims,

                                           dtype="float",compression='gzip',compression_opts=9)

        self.labels = self.db.create_dataset("labels", (dims[0],),

                                             dtype="int",compression='gzip',compression_opts=9)

        self.bufSize = bufSize

        self.buffer = {"data": [], "labels": []}

        self.idx = 0



    def add(self, rows, labels):



        # add the rows and labels to the buffer

        self.buffer["data"].extend(rows)

        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:

            self.flush()



    def flush(self):



        # write the buffers to disk then reset the buffer

        i = self.idx + len(self.buffer["data"])

        self.data[self.idx:i] = self.buffer["data"]

        self.labels[self.idx:i] = self.buffer["labels"]

        self.idx = i

        self.buffer = {"data": [], "labels": []}



    def storeClassLabels(self, classLabels):



        # create a dataset to store the actual class label names,

        # then store the class labels

        dt = h5py.special_dtype(vlen=str)

        labelSet = self.db.create_dataset("label_names",

                                          (len(classLabels),), dtype=dt)

        labelSet[:] = classLabels



    def close(self):



        # check to see if there are any other entries in the buffer

        # that need to be flushed to disk

        if len(self.buffer["data"]) > 0:

            self.flush()



        # close the dataset

        self.db.close()
from hdf5datasetwriter import HDF5DatasetWriter

from tqdm import tqdm



for (images,labels,outputPath) in tqdm(datasets):

    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    for image,label in zip(images,labels):

        writer.add([image],[label])

    

    writer.close()



f.close()

    
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D,MaxPooling2D

from keras.layers.advanced_activations import ELU

from keras.layers.core import Activation,Flatten,Dropout,Dense

from keras import backend as K

height=48

width=48

depth=1

model=Sequential()

inputShape = (height, width, depth)

chanDim = -1



model.add(Conv2D(32, (3, 3), padding="same",kernel_initializer="he_normal", input_shape=inputShape))

model.add(Activation('relu'))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",

padding="same"))

model.add(Activation('relu'))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",

padding="same"))

model.add(Activation('relu'))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",

padding="same"))

model.add(Activation('relu'))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",

padding="same"))

model.add(Activation('relu'))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",

padding="same"))

model.add(ELU())

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64, kernel_initializer="he_normal"))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer="he_normal"))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(7, kernel_initializer="he_normal"))

model.add(Activation("softmax"))
%%writefile hdf5datasetgenerator.py

from keras.utils import np_utils

import numpy as np

import h5py



class HDF5DatasetGenerator:

    def __init__(self, dbPath, batchSize, preprocessors=None,

    aug=None, binarize=True, classes=2):

        # store the batch size, preprocessors, and data augmentor,

        # whether or not the labels should be binarized, along with

        # the total number of classes

        self.batchSize = batchSize

        self.preprocessors = preprocessors

        self.aug = aug

        self.binarize = binarize

        self.classes = classes



        # open the HDF5 database for reading and determine the total

        # number of entries in the database

        self.db = h5py.File(dbPath)

        self.numImages = self.db["labels"].shape[0]



    def generator(self, passes=np.inf):

        # initialize the epoch count

        epochs = 0

        while epochs < passes:

            # loop over the HDF5 dataset

            for i in np.arange(0, self.numImages, self.batchSize):

                # extract the images and labels from the HDF dataset

                images = self.db["images"][i: i + self.batchSize]

                labels = self.db["labels"][i: i + self.batchSize]

                if self.binarize:

                    labels = np_utils.to_categorical(labels,

                                                     self.classes)



                # check to see if our preprocessors are not None

                if self.preprocessors is not None:

                    # initialize the list of processed images

                    procImages = []



                    for image in images:

                        # loop over the preprocessors and apply each

                        # to the image

                        for p in self.preprocessors:

                            image = p.preprocess(image)

                        procImages.append(image)

                    images = np.array(procImages)



                if self.aug is not None:

                    (images, labels) = next(self.aug.flow(images,

                                                          labels, batch_size=self.batchSize))

                yield (images, labels)



            epochs += 1



    def close(self):

        # close the datab

        self.db.close()
%%writefile imagetoarraypreprocessor.py

from keras.preprocessing.image import img_to_array



class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None):

        # store the image data format

        self.dataFormat = dataFormat



    def preprocess(self, image):

        # apply the Keras utility function that correctly rearranges

        # the dimensions of the image

        return img_to_array(image, data_format=self.dataFormat)
from hdf5datasetgenerator import HDF5DatasetGenerator

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from imagetoarraypreprocessor import ImageToArrayPreprocessor
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,

horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

iap = ImageToArrayPreprocessor()



trainGen = HDF5DatasetGenerator('/kaggle/working/train', 128,

aug=trainAug,preprocessors=[iap],  classes=7)

valGen = HDF5DatasetGenerator('/kaggle/working/validation', 128,

aug=valAug,preprocessors=[iap], classes=7)





opt = Adam(lr=1e-3)

model.compile(loss="categorical_crossentropy", optimizer=opt,

metrics=["accuracy"])
model.fit_generator(

trainGen.generator(),

steps_per_epoch=trainGen.numImages // 128,

validation_data=valGen.generator(),

validation_steps=valGen.numImages // 128,

epochs=50,verbose=1)





trainGen.close()

valGen.close()
model.save('model.hdf5')