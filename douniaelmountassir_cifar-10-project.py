

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

from keras.callbacks import ModelCheckpoint, EarlyStopping





from keras.optimizers import adam

from keras.callbacks import Callback

from keras.utils import np_utils #to transform labels in categorical



from keras import backend as K

 #To tell Tensorflow the right order of dims

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.datasets import cifar10

import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.shape
y_train.shape
x_test.shape
plt.imshow(x_test[8])
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
mean = np.mean(x_train, axis = (0,1,2,3))

std = np.std(x_train, axis = (0,1,2,3))

x_train = (x_train-mean)/(std+1e-7)

x_test = (x_test-mean)/(std+1e-7)
np.std(x_train, axis = (0,1,2,3))
nClasses = 10

y_train = np_utils.to_categorical(y_train,nClasses)

y_test = np_utils.to_categorical(y_test,nClasses)
print(x_train.shape)

print(y_train.shape)
input_shape = (32,32,3)
def createModel():

    

    model = VGG16(include_top=False, weights='imagenet',input_shape=(32,32,3)) #VGG16 pour avoir mieux résultats



    myModel = Sequential()

    for layer in model.layers:

        myModel.add(layer)



    myModel.add(Flatten())

    myModel.add(Dense(512, activation='relu'))

    myModel.add(Dropout(0.5))

    myModel.add(Dense(10, activation='softmax')) 

    

    return myModel
K.clear_session()

model = createModel()
AdamOpt = adam(lr=0.001)

model.compile(optimizer=AdamOpt, loss='categorical_crossentropy', metrics=['accuracy'])
# Pour augmenter les cas de training afin d'avoir de mieux résultats on utilise (Data augmentation)

datagen = ImageDataGenerator(

    

    featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        zca_epsilon=1e-06,  # epsilon for ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        # randomly shift images horizontally (fraction of total width)

        width_shift_range=0.1,

        # randomly shift images vertically (fraction of total height)

        height_shift_range=0.1,

        shear_range=0.1,  # set range for random shear

        zoom_range=0.1,  # set range for random zoom

        channel_shift_range=0.,  # set range for random channel shifts

        # set mode for filling points outside the input boundaries

        fill_mode='nearest',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

        # set rescaling factor (applied before any other transformation)

        rescale=None,

        # set function that will be applied on each input

        preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

        data_format=None,

        # fraction of images reserved for validation (strictly between 0 and 1)

        validation_split=0.0)

# (std, mean, and principal components if ZCA whitening is applied).
model.summary() #Description of the model
batch_size = 256

epochs = 1000



checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

#ErlyStoping pour éviter le Surapprentissage



history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // batch_size,

epochs=epochs,callbacks=[checkpoint,early])

# model.fit_generator pour commencé et pour avoir mieux résultats en utilisant vgg16
plt.figure()

plt.plot(history.history['loss'],'blue',linewidth=3.0)

plt.plot(history.history['val_loss'],'red',linewidth=3.0)

plt.legend(['Training loss','Validation loss'],fontsize=18)

plt.xlabel('Epochs',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

plt.show()
plt.figure()

plt.plot(history.history['accuracy'],'blue',linewidth=3.0)

plt.plot(history.history['val_accuracy'],'red',linewidth=3.0)

plt.legend(['Training accuracy','Validation accuracy'],fontsize=18,loc='lower right')

plt.xlabel('Epochs',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)