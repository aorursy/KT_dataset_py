# importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import patches

#import matplotlib.pyplot.axis(*args, **kwargs)

import os

print(os.listdir("../input"))
# read the csv file using read_csv function of pandas

train_df = pd.read_csv('../input/train.csv')



train_df.head()
# reading single image using imread function of matplotlib

#im_0 = Image(filename ='../input/train_images/002c21358ce6.png')



image0 = plt.imread('../input/train_images/000c1434d8d7.png')

image1 = plt.imread('../input/train_images/001639a390f0.png')

image2 = plt.imread('../input/train_images/0024cdab0c1e.png')

image3 = plt.imread('../input/train_images/002c21358ce6.png')

image4 = plt.imread('../input/train_images/005b95c28852.png')

plt.imshow(image0)

xmin, xmax, ymin, ymax = plt.axis()

print(xmin,xmax,ymin,ymax)
plt.imshow(image1)
plt.imshow(image2)
plt.imshow(image3)
plt.imshow(image4)
# Number of unique training images

train_df['id_code'].nunique()
# Number of classes

train_df['diagnosis'].value_counts()
train_df = pd.read_csv('../input/train.csv')

train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['id_code'] = train_df['id_code'].astype(str)+'.png'
from keras.preprocessing.image import ImageDataGenerator



datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.2)



batch_size = 32



train_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(96,96),

    subset='training')



test_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    target_size=(96,96),

    subset='validation')
y_train = train_df['diagnosis']

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, GaussianNoise, GaussianDropout

from keras.layers import Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, SeparableConv2D

from keras.constraints import maxnorm

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

from keras import regularizers, optimizers
def build_model():

    # create model

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=[96,96,3], activation='relu'))

    model.add(GaussianDropout(0.3))

    model.add(Conv2D(32, (5, 5), activation='relu', kernel_constraint=maxnorm(3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(Conv2D(128, (7, 7), activation='relu'))

    

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)

                   ,activity_regularizer=regularizers.l1(0.01)))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(lr=0.002), metrics=['accuracy'])

    print('compiling model with optimizer Adamax and using filters 32,64,128')

    return model

#Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#original

#model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, Adamax=True), metrics=['accuracy'])

#model.compile(optimizer='rmsprop',              loss='categorical_crossentropy',              metrics=['accuracy'])

model=build_model()
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint

es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 0, patience = 2)

mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 0)
Num_folds=3

for i in range(Num_folds):

    print("Training on Fold: ",i+1)



    model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen),

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen),

                                    epochs=3,

                                    callbacks = [es, mc], 

                                    use_multiprocessing = True,

                                    verbose=1)

  
#Due to the timeconstraints here for implementation we have limited epoch =3,(use epoch=20) folds=3(use fold more than this) 
from keras.models import load_model

model = load_model('model.h5')
model.summary()
submission_df = pd.read_csv('../input/sample_submission.csv')



submission_df['id_code'] = submission_df['id_code'].astype(str)+'.png'
submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission_df,

    directory="../input/test_images",

    x_col="id_code",    

    batch_size=batch_size,

    shuffle=False,

    class_mode=None, 

    target_size=(96,96)

)
predictions=model.predict_generator(submission_gen, steps = len(submission_gen))
max_probability = np.argmax(predictions,axis=1) 
submission_df['diagnosis'] = max_probability

submission_df.to_csv('submission.csv', index=False)