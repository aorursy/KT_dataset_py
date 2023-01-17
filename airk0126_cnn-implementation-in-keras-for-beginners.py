import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from keras.utils import np_utils
# data attribute

ROOT_PATH = '../input/Kannada-MNIST/'

class_num = 10

width = 28

height = 28

color_num = 1

input_shape = (width, height, color_num)
# load data

train = pd.read_csv(ROOT_PATH+'train.csv').values

dig = pd.read_csv(ROOT_PATH+'Dig-MNIST.csv').values

X_test = pd.read_csv(ROOT_PATH+'test.csv').drop('id', axis=1).values
# reshape train data

y = train[:, 0].astype('int32')

X = train[:, 1:].astype('float32').reshape(-1, width, height, color_num)



# train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)



# reshape check data

y_dig = dig[:, 0].astype('int32')

X_dig = dig[:, 1:].astype('float32').reshape(-1,  width, height, color_num)



# reshape test data

X_test = X_test.astype('float32').reshape(-1,  width, height, color_num)
# plot training images

fig = plt.figure(figsize=(15, 15))



for i in range(class_num):

    ax = fig.add_subplot(2, class_num, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(width, height), cmap='gray')

    ax.set_title(str(y_train[i]))
# one-hot encode the labels

y_train = np_utils.to_categorical(y_train, class_num)

y_val = np_utils.to_categorical(y_val, class_num)



# setting keras input data

train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range = 10,

                                   width_shift_range = 0.3,

                                   height_shift_range = 0.3,

                                   shear_range = 0.2,

                                   zoom_range = 0.3)



valid_datagen = ImageDataGenerator(rescale=1./255)
# define the model

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
# set parameter

batch_size = 1024

epochs = 30



# Set a learning rate annealer

lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# model compile

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999),

              metrics=['accuracy'])



# model fitting

history = model.fit_generator(  train_datagen.flow(X_train, y_train, batch_size=batch_size),

                                steps_per_epoch=epochs,

                                epochs=epochs,

                                workers=4,

                                callbacks=[lr_reducer],

                                validation_data=valid_datagen.flow(X_val, y_val),

                                validation_steps=epochs/2,

                                verbose=2)
# Check accuracy 

preds_dig = model.predict_classes(X_dig/255, verbose=2)



print(metrics.accuracy_score(preds_dig, y_dig))
# predict test data

testY = model.predict_classes(X_test, verbose=2)



# output

sub = pd.read_csv(ROOT_PATH+'sample_submission.csv')

sub['label'] = testY

sub.to_csv('submission.csv', index=False)
# check submisson.csv

sub.head()