%matplotlib inline

import keras

from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.preprocessing import image

from PIL import Image



from numpy import array

import numpy as np



import pandas as pd



from matplotlib import pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit



batch_size = 64;

dim = 20;
trainRows = pd.read_csv('../input/train.csv')

testRows = pd.read_csv('../input/test.csv')
xAll = np.empty((len(trainRows['id']), dim, dim, 1)) 

print(xAll.shape)

i = 0

for id in trainRows['id']:

    filename = "../input/images/"+str(id)+".jpg"

    im = Image.open(filename)

    im.thumbnail([dim, dim])

    im = array(im)

    height, width = im.shape

    

    #calculate destination coordinates

    h1 = int((dim - height) / 2)

    h2 = h1 + height

    w1 = int((dim - width) / 2)

    w2 = w1 + width

    

    xAll[i, h1:h2, w1:w2, 0] = im

    i += 1

    

print(xAll.shape)

yAll = trainRows.pop('species')

yAll = LabelEncoder().fit(yAll).transform(yAll)

yAll = to_categorical(yAll)

yAll.shape

trainRows.pop('id')

featuresAll = StandardScaler().fit(trainRows).transform(trainRows)
#start with a simple FC layer



meanX = xAll.mean().astype(np.float32)

stdX = xAll.std().astype(np.float32)



def normalize(x): 

    return (x-meanX)/stdX



def enrich(x):

    #take a flattened image array and add the additional parameters

    return ('foo')



model = Sequential([

    Lambda(normalize, input_shape=(dim,dim,1)),

        Convolution2D(16,3,3, activation='elu'),

        Convolution2D(16,3,3, activation='elu'),

        MaxPooling2D(),

        Convolution2D(32,3,3, activation='elu'),

        ZeroPadding2D((1, 1)),

        Convolution2D(32,3,3, activation='elu'),

        MaxPooling2D(),

        Flatten(),

        Dense(20, activation='elu'),

        Dense(yAll.shape[1], activation='softmax')

    ])

model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#split test and validation

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=0)

trainIndex, valIndex = next(sss.split(xAll, yAll))

xTrain, yTrain = xAll[trainIndex], yAll[trainIndex]

featuresTrain = featuresAll[trainIndex]

xVal, yVal = xAll[valIndex], yAll[valIndex]

featuresVal = featuresAll[valIndex]
print(xTrain.shape)

print(yTrain.shape)

print(xVal.shape)

print(yVal.shape)

print(featuresTrain.shape)

print(featuresVal.shape)
generator = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 

                                     height_shift_range=0.08, zoom_range=0.08)

trainBatches = generator.flow(xTrain, yTrain, batch_size=batch_size)

valBatches = generator.flow(xVal, yVal, batch_size=batch_size)
model.fit_generator(trainBatches, trainBatches.n, nb_epoch=1, 

                    validation_data=valBatches, nb_val_samples=valBatches.n)