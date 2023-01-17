from sklearn.datasets import load_files

import numpy as np



train_dir = '../input/fruits/fruits-360_dataset/fruits-360/Training'

test_dir = '../input/fruits/fruits-360_dataset/fruits-360/Test'
import numpy as np

import cv2

import os

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):

        # store the image preprocessor

        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an

        # empty list

        if self.preprocessors is None:

            self.preprocessors = []





    def load(self, imagePaths, verbose=-1):

        # initialize the list of features and labels

        data = []

        labels = []

        # loop over the input images

        for (i, imagePath) in enumerate(imagePaths):

            # load the image and extract the class label assuming

            # that our path has the following format:

            # /path/to/dataset/{class}/{image}.jpg

            image = cv2.imread(imagePath)

            label = imagePath.split(os.path.sep)[-2]



            if self.preprocessors is not None:

                for p in self.preprocessors:

                    image = p.preprocess(image)

                    # treat our processed image as a "feature vector"

                    # by updating the data list followed by the labels

            data.append(image)

            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:

                print("[INFO] processed {}/{}".format(i + 1,

                len(imagePaths)))

                # return a tuple of the data and labels

        return (np.array(data), np.array(labels))
import cv2



class SimplePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):

        # store the target image width, height, and interpolation

        # method used when resizing

        self.width = width

        self.height = height

        self.inter = inter



    def preprocess(self, image):

        # resize the image to a fixed size, ignoring the aspect

        # ratio

        return cv2.resize(image, (self.width, self.height),

        interpolation=self.inter)
!pip install imutils

import imutils

from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array



class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None):

        # store the image data format

        self.dataFormat = dataFormat

    def preprocess(self, image):

        # apply the Keras utility function that correctly rearranges

        # the dimensions of the image

        return img_to_array(image, data_format=self.dataFormat)
train_imagePaths = list(paths.list_images(train_dir))

np.random.shuffle(train_imagePaths)

test_imagePaths = list(paths.list_images(test_dir))

np.random.shuffle(test_imagePaths)



sp = SimplePreprocessor(32,32)

iap = ImageToArrayPreprocessor()



dataloader = SimpleDatasetLoader(preprocessors = [sp, iap])

trainData, trainLabels = dataloader.load(train_imagePaths)

testData, testLabels = dataloader.load(test_imagePaths)
no_of_classes = len(np.unique(trainLabels))

print('Number of classes', no_of_classes)

print('Training set size', trainData.shape[0])

print('Testing set size', testData.shape[0])
trainData = trainData.astype('float32')/255.0

testData = testData.astype('float32')/255.0
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

trainLabels = lb.fit_transform(trainLabels)

testLabels = lb.fit_transform(testLabels)
print(trainLabels[0:10])
print('Training set shape', trainData.shape)

print('Testing set shape', testData.shape)
trainData[0]
import matplotlib.pyplot as plt

import cv2

fig = plt.figure(figsize = (30,5))

for i in range(10):

    ax = fig.add_subplot(2,5,i+1,xticks = [], yticks = [])

    image = cv2.cvtColor(trainData[i], cv2.COLOR_BGR2RGB)

    ax.imshow(np.squeeze(image))
import matplotlib.image as mpimg



image = mpimg.imread(train_imagePaths[1])

image = imutils.resize(image, width = 32, inter = cv2.INTER_CUBIC)

plt.imshow(image)

imgplot = plt.imshow(image)

plt.axis('off')

plt.show()



labelNames = np.array([train_imagePath.split(os.path.sep)[-2] for train_imagePath in train_imagePaths])

print(labelNames[1])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense



class MiniVGGNet:

    @staticmethod

    def build(width, height, depth, classes):

        # initialize the model along with the input shape to be

        # "channels last" and the channels dimension itself

        model = Sequential()

        inputShape = (height, width, depth)



        # first CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(32, (3, 3), padding="same",

        input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(32, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # second CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))



        # softmax classifier

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        # return the constructed network architecture

        return model
from tensorflow.keras.optimizers import SGD

opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=no_of_classes)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath = 'minivggnet_fruits.hdf5', verbose = 1, save_best_only = True)
H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=64, callbacks = [checkpointer], epochs=10, verbose=1)
model.load_weights('minivggnet_fruits.hdf5')
labelNames = np.array([test_imagePath.split(os.path.sep)[-2] for test_imagePath in test_imagePaths])
print(np.unique(labelNames))
from sklearn.metrics import classification_report



predictions = model.predict(testData, batch_size=64)

print(classification_report(testLabels.argmax(axis=1),predictions.argmax(axis=1), target_names=np.unique(labelNames)))
import cv2



fig = plt.figure(figsize = (16,9))

np.random.seed(42)

for i, idx in enumerate(np.random.choice(testData.shape[0], size = 16, replace = False)):

    ax = fig.add_subplot(4,4,i+1,xticks = [],yticks = [])

    image = cv2.cvtColor(testData[i], cv2.COLOR_BGR2RGB)

    ax.imshow(np.squeeze(image))

    pred_idx = np.argmax(predictions[idx])

    true_idx = np.argmax(testLabels[idx])

    ax.set_title('{} ({})'.format(labelNames[pred_idx], labelNames[true_idx]), color = ('green' if pred_idx == true_idx else 'red'))

    
import matplotlib.pyplot as plt



plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")



plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on Fruits-360 dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()