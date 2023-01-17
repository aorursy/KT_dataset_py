# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
def getDigitsAndPixelRows(train):

    digits=train['label']

    pixelrows=train.drop(['label'],axis=1)

    return digits,pixelrows    
def reshapePixels(pixelrows,test):

    pixelrows=pixelrows.values.reshape(-1,28,28,1)

    test=test.values.reshape(-1,28,28,1)

    pixelrows=pixelrows/255

    test=test/255

    return pixelrows,test
digits,pixelrows=getDigitsAndPixelRows(train)

pixelrows,test=reshapePixels(pixelrows,test)
digits=to_categorical(digits,num_classes=10)
def createModel():

    model=Sequential()

    

    model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu',input_shape=(28,28,1),padding='same'))

    model.add(BatchNormalization())

        

    model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu',padding='same'))

    model.add(BatchNormalization())    

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(.25))

    

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Dropout(.25))

    

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(Dropout(.25))

    

    model.add(Flatten())

    model.add(Dense(units=256,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(.25))

    

    model.add(Dense(units=10,activation='softmax'))

    

    return model    
def compileModel(model):

    optimizer=RMSprop(lr=0.001,rho=.9,epsilon=1e-08,decay=0.0)

    lossFunction='categorical_crossentropy'

    metrics=['accuracy']

    model.compile(optimizer=optimizer,loss=lossFunction,metrics=metrics)    

    return model
model=compileModel(createModel())
learningRateReduction=ReduceLROnPlateau(monitor='val_acc',

                                       factor=.5,

                                       patience=3,

                                       verbose=1,

                                       min_lr=0.00001)
xpixels,xdigits,ypixels,ydigits=train_test_split(pixelrows,digits,test_size=.1,random_state=2)
datagen = ImageDataGenerator(

        featurewise_center=False, # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(xpixels)
 

noOfEpochs=1

batchSize=128

    

history=model.fit_generator(generator=datagen.flow(xpixels,ypixels,batch_size=batchSize),

                        steps_per_epoch=xpixels.shape[0],

                        epochs=noOfEpochs,

                        verbose=2,

                        callbacks=[learningRateReduction],

                        validation_data=(xdigits,ydigits),

                        validation_steps=None,                       

                        class_weight=None,

                        max_queue_size=10,

                        workers=1,

                        use_multiprocessing=False,

                        shuffle=True,

                        initial_epoch=0

                    )

results=model.predict(test)

results=np.argmax(results,axis=1)

results=pd.Series(results,name="Label")

submission=pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)

submission.to_csv("DigitRecogsubmission.csv",index=False)
print(submission.info())