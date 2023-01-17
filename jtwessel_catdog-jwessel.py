# Joseph Wessel

# Last Modified: March 3, 2017

# Data Science



# NOTE: This code is heavily based on, and consists almost exclusively of, modified code provided

#       in George Mohler's kernel found here:

#       https://www.kaggle.com/george04/dogs-vs-cats-redux-kernels-edition/cnn-cat-dog/code



import numpy as numpy

import pandas as pandas



from subprocess import check_output



import os,cv2,random



from keras.models import Sequential

from keras.layers import Input,Dropout,Flatten,Convolution2D,MaxPooling2D,Dense,Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping

from keras.utils import np_utils



# Add references to test and training file directories:



trainDir = '../input/train/'

testDir = '../input/test/'



# Specify 32 rows and columns, and only one channel, which simplifies the 

# processing (vs. 64 rows and columns, and 3 channels):



rows = 32

cols = 32

channels = 1



# Create arrays of training and testing images:



trainImages = [trainDir+i for i in os.listdir(trainDir)]

testImages = [testDir+i for i in os.listdir(testDir)]



# Define functions for reading in the image and preparing the image for processing:



def readImage(filePath):

    img = cv2.imread(filePath,cv2.IMREAD_GRAYSCALE)

    return cv2.resize(img,(rows,cols),interpolation=cv2.INTER_CUBIC)



def prepData(images):

    count = len(images)

    data = numpy.ndarray((count,channels,rows,cols),dtype=numpy.uint8)

    

    for i,imageFile in enumerate(images):

        image = readImage(imageFile)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i,count))

    

    return data



# Implement training and data sets, and append labels for submission:



train = prepData(trainImages)

test = prepData(testImages)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))



labels = []

for i in trainImages:

	if 'dog' in i:

		labels.append(1)

	else:

		labels.append(0)



# Train on image shape:        

        

train = train.reshape(-1,32,32,1)

test = test.reshape(-1,32,32,1)

xTrain = train.astype('float32')

xTest = test.astype('float32')

xTrain /= 255

xTest /= 255

yTrain = labels



xValid = xTrain[:5000,:,:,:]

yValid = yTrain[:5000]

xTrain = xTrain[5001:25000,:,:,:]

yTrain = yTrain[5001:25000]



print("Training matrix shape",xTrain.shape)

print("Training matrix shape",xTest.shape)



# Fit to model:



model = Sequential()

model.add(Convolution2D(16,3,3,border_mode='same',input_shape=(rows,cols,channels),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(100,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



model.fit(xTrain,yTrain,

		  batch_size = 128, nb_epoch = 8,

		  show_accuracy = True, verbose = 1,

		  validation_data = (xValid,yValid))



# Prepare submission CSV file:



submission = model.predict_proba(xTest, verbose = 1)

testID = range(1,12501)

predictionsDF = pandas.DataFrame({'id': testID, 'label': submission[:,0]})



predictionsDF.to_csv("JoeWessel_Submissions.csv", index = False)