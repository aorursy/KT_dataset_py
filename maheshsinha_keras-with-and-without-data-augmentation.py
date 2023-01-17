!pip install imutils

from keras.preprocessing.image import img_to_array

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Flatten, Dropout, Activation

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.optimizers import SGD

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import imutils

import cv2

import tarfile

import shutil

import os





class AspectAwarePreprocessor():

    def __init__(self, width, height, inter = cv2.INTER_AREA):

        self.height = height

        self.width = width

        self.inter = inter

        

    def preprocess(self, image):

        #grab the dimensions of the image and then initialize the deltas to use when cropping:

        (h, w) = image.shape[:2]

        dW = 0

        dH = 0

        

        #if width is smaller than height, resize along the width (i.e. smaller dimension)..

        #.. then update the deltas to crop the height to the desired dimension.// 

        #//Otherwise if the height is smaller than the width, resize along the height..

        #..and update the deltas of width

        if w<h:

            image = imutils.resize(image, width=self.width, inter=self.inter)

            dH = int((image.shape[0]-self.height)/2.0)

            

        else:

            image = imutils.resize(image, height=self.height, inter=self.inter)

            dW = int((image.shape[1]-self.width)/2.0)

            

            

        #Now that we have resiezed the image, we need to regrab the height and width, ..

        #.. followed by performing the corp

        (h, w) = image.shape[:2]

        image = image[dH:h-dH, dW:w-dW]

        

        #finally resize the image with provided spatial dimensions..

        #..to ensure the output image is alway fixed size:

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

    

class ImageToArrayPreprocessor:

    def __init__(self, data_format=None):

        self.data_format = data_format

    

    def preprocess(self, image):

        return img_to_array(image, data_format=self.data_format)

        

class SimpleDatasetLoader():

    def __init__(self, preprocessors=None):

        """

        :param preprocessors: List of image preprocessors

        """

        self.preprocessors = preprocessors

        if self.preprocessors == None:

            self.preprocessors = []

    

    # Method: Used to load a list of images for pre-processing

    def load(self, imagePaths, verbose=-1):

        data = []

        labels = []

        for i, imagePath in enumerate(imagePaths):

            image = cv2.imread(imagePath)

            label = imagePath.split('/')[-2]



            if self.preprocessors is not None:

                for p in self.preprocessors:

                    image = p.preprocess(image)

            data.append(image)

            labels.append(label)

            

            if verbose>0 and (i+1)%verbose==0:

                print(f'[INFO]: Processed {i+1}/{len(imagePaths)}')

        

        return (np.array(data), np.array(labels))



class MiniVGGNet:

    @staticmethod

    def build(width, height, depth, classes):

        #intialize the model, inputShape and channelDimensions:

        model = Sequential()

        inputShape = (width, height, depth)

        channel_dim = -1

        

        #if we are using the "channel_first", update the inputShape and channel_dim:

        if K.image_data_format == 'channel_first':

            inputShape = (depth, height, width)

            channel_dim = 1

            

        # First CONV => RELU => CONV => RELU => POOL layer set:

        model.add(Conv2D(32, (3,3), padding='same', input_shape = inputShape))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(32, (3,3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        

        #Second CONV => RELU => CONV => RELU => POOL layer set:

        model.add(Conv2D(64, (3,3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(64, (3,3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        

        #First set of FC => RELU:

        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=channel_dim))

        model.add(Dropout(0.5))

        

        #Softmax Classifier:

        model.add(Dense(classes))

        model.add(Activation('softmax'))

        

        return model





if not os.path.exists('/kaggle/working/flower17_dataset/'):

    print('[INFO]: Extracting and Preparing data directory structure...')

    #Extracting all images from tgz comprissed file, which extracts all images in a single folder "jpg"

    tar = tarfile.open('/kaggle/input/17-category-flowers/17flowers.tgz')

    tar.extractall()



    #Forimg data directory structure:

    j = 1

    total = 1361

    for i  in range(1, total):

        fpath = f"jpg/image_{str(i).zfill(4)}.jpg"

        destPath = 'flower17_dataset/'+str(j)

        if not os.path.exists(destPath):

            os.makedirs(destPath)

        shutil.copy(fpath, destPath)



        if i%80==0:

            j+=1
dataset_path = '/kaggle/working/flower17_dataset/'



#grab the list of images, and label from imagePaths:

print('[INFO] loading image...')

imagePaths = [i.replace('\\', '/') for i in list(paths.list_images(dataset_path))]



#initialize the image preprocessors:

aap = AspectAwarePreprocessor(64, 64)

iap = ImageToArrayPreprocessor()



#load the dataset from the disk and scale the raw piel intensities to range [0,1]:

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])

(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.astype('float')/255.0



# partition the data into training and testing splits using 75% of

# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)



#convert the labels from int to vectors:

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)



#intialize the optimizer and model:

print('[INFO] optimizing model...')

opt = SGD(lr=0.05)



model = MiniVGGNet.build(64, 64, 3, len(np.unique(labels)))

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics =['accuracy'])



#train the network:

print('[INFO] training network...')

H = model.fit(trainX, trainY, epochs = 100, validation_data=(testX, testY), batch_size=32, verbose=1)



#evaluate the network:

print('[INFO] evaluating network...')

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))



# plot the training loss and accuracy

plt.figure()

plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
dataset_path = '/kaggle/working/flower17_dataset/'



#grab the list of images, and label from imagePaths:

print('[INFO] loading image...')

imagePaths = [i.replace('\\', '/') for i in list(paths.list_images(dataset_path))]



#initialize the image preprocessors:

aap = AspectAwarePreprocessor(64, 64)

iap = ImageToArrayPreprocessor()



#load the dataset from the disk and scale the raw piel intensities to range [0,1]:

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])

(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.astype('float')/255.0



# partition the data into training and testing splits using 75% of

# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)



#convert the labels from int to vectors:

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)



#construct the data generator for image augmentation:

aug = ImageDataGenerator(rotation_range=30, 

                         width_shift_range=0.1, 

                         height_shift_range=0.1, 

                         shear_range=0.2, zoom_range=0.2, 

                         horizontal_flip=True, 

                         fill_mode='nearest')



#intialize the optimizer and model:

print('[INFO] optimizing model...')

opt = SGD(lr=0.05)



model = MiniVGGNet.build(64, 64, 3, len(np.unique(labels)))

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics =['accuracy'])



#train the network with augmentation:

print('[INFO] training network with data augmentation...')

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), 

                        epochs = 100, 

                        steps_per_epoch=len(trainX)//32, 

                        validation_data=(testX, testY), 

                        verbose=1)



#evaluate the network:

print('[INFO] evaluating network...')

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))



# plot the training loss and accuracy

plt.figure()

plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()