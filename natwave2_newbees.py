# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from glob import glob

from random import shuffle

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten

from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D

from keras.layers.pooling import _GlobalPooling1D

from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

from keras.models import Model

from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from imgaug import augmenters as iaa

import imgaug as ia #not from keras, from other image aug library

folderLoc = "../input"

print(os.listdir(folderLoc))# listing of the names of the files and folder in folderloc

trainData = pd.read_csv(folderLoc+"/train_labels.csv")

idLabel = {col1:col2 for col1,col2 in zip(trainData.id.values, trainData.label.values)}# to map between columns abd labels

trainData.head()# to display all the information in the files
def getId(filePath):# get the location and give all the tif format files

    return filePath.split(os.path.sep)[-1].replace('.tif', '')# find all tif files and return 
trainFiles = glob(folderLoc+'/train/*.tif')# find all train TIF files and place in the variable 

testFiles = glob(folderLoc+'/test/*.tif') # find all test TIF files and place in the variable

print("labeled_files size :", len(trainFiles))# lenght of Train/labeled file

print("test_files size :", len(testFiles)) # lenght of test file
# split the train file into 95% train and 5% val

trainSet, validationSet = train_test_split(trainFiles, test_size=0.05, random_state=123456)

def getChunk(sequence, seqSize):# agreegate sequences with some size, chunk and return

    return (sequence[position:position + seqSize] for position in range(0, len(sequence), seqSize))


def getSequence():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug) # flip negative 0.5% sometimes

    seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)

                rotate=(-10, 10), # rotate by -45 to +45 degrees

                shear=(-5, 5), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-1, 0),

                            first=iaa.Multiply((0.9, 1.1), per_channel=True),

                            second=iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True

    )

    return seq
def dataGeneration(listFiles, idLabel, batchSize, augment=False):

    seq = getSequence()

    while True:

        shuffle(listFiles) # shuffle all files from the folder

        for batch in getChunk(listFiles, batchSize): # read the files as a chunk based on size and send it to a batch

            X = [cv2.imread(x) for x in batch] # X is collection of images from that batch

            Y = [idLabel[getId(x)] for x in batch] # Y is labels getting from the train_label.csv file

            if augment:

                X = seq.augment_images(X)

            X = [preprocess_input(x) for x in X]

                

            yield np.array(X), np.array(Y)
def getModel():

    inputs = Input((96, 96, 3)) # input size based keras implementation

    base_model = NASNetMobile(include_top=False, input_shape=(224, 224, 3))#input_shape should be 128, 160, 192, 224

    x = base_model(inputs) # base model from keras library

    out1 = GlobalMaxPooling2D()(x)

    out2 = GlobalAveragePooling2D()(x)

    out3 = Flatten()(x)

    out = Concatenate(axis=-1)([out1, out2, out3])

    out = Dropout(0.5)(out)

    out = Dense(1, activation="sigmoid", name="3_")(out)

    model = Model(inputs, out)

    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])

    model.summary()



    return model
model = getModel()
batchSize=32

h5Path = "model.h5"

checkpoint = ModelCheckpoint(h5Path, monitor='val_acc', verbose=2, save_best_only=True, mode='max')



history = model.fit_generator(

    dataGeneration(trainSet, idLabel, batchSize, augment=True),

    validation_data=dataGeneration(validationSet, idLabel, batchSize),

    epochs=2, verbose=1,

    callbacks=[checkpoint],

    steps_per_epoch=len(trainSet) // batchSize,

    validation_steps=len(validationSet) // batchSize)

batchSize=64

history = model.fit_generator(

    dataGeneration(trainSet, idLabel, batchSize, augment=True),

    validation_data=dataGeneration(validationSet, idLabel, batchSize),

    epochs=6, verbose=1,

    callbacks=[checkpoint],

    steps_per_epoch=len(trainSet) // batchSize,

    validation_steps=len(validationSet) // batchSize)



model.load_weights(h5Path)
predictions = []

ids = []

for batch in getChunk(testFiles, batchSize):

    X = [preprocess_input(cv2.imread(x)) for x in batch]

    idsBatch = [getId(x) for x in batch]

    X = np.array(X)

    predsBatch = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    predictions += predsBatch

    ids += idsBatch
#Trian and validation loss graph

val_loss = history.history['val_loss']

loss = history.history['loss']

plt.plot(range(len(val_loss)),val_loss,'c',label='Validation loss')

plt.plot(range(len(loss)),loss,'m',label='Train loss')



plt.title('Training and validation losses')

plt.legend()

plt.xlabel('epochs')

plt.show()
dataFrame = pd.DataFrame({'id':ids, 'label':predictions})

dataFrame.to_csv("result.csv", index=False)

dataFrame.head()