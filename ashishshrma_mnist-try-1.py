# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

#import seaborn as sns

#import matplotlib inline



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.optimizers import Adam 

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



y_train = train["label"]



# Drop 'label' column

x_train = train.drop(labels = ["label"],axis = 1) 



y_train.value_counts()
# Normalize the data

x_train = x_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
print(x_train.shape)
random_seed = 2
# Split the train and the validation set for the fitting

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
# change our image type to float32 data type

#x_train = x_train.astype('float32')

#x_test = x_test.astype('float32')







print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_val.shape[0], 'test samples')



# Now we one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_val = np_utils.to_categorical(y_val)



# Let's count the number columns in our hot encoded matrix 

print ("Number of Classes: " + str(y_val.shape[1]))



num_classes = y_val.shape[1]

num_pixels = x_train.shape[1] * x_train.shape[2]



# create model

model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 padding = "same",

                 input_shape=(28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 padding = "same",

                 input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#Dense Layers



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



#Final Dense Layer

model.add(Dense(num_classes, activation='softmax'))



optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



model.compile(loss = 'categorical_crossentropy',

              optimizer = optimizer,

              metrics = ['accuracy'])



print(model.summary())

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

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





datagen.fit(x_train)
batch_size = 128

epochs = 50



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement

                          min_delta = 0, #Abs value and is the min change required before we stop

                          patience = 5, #Number of epochs we wait before stopping 

                          verbose = 1,

                          restore_best_weights = True) #keeps the best weigths once stopped



callbacks = [earlystop, learning_rate_reduction]



history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

          epochs=epochs,

          verbose=1,

          callbacks=callbacks,

          steps_per_epoch = x_train.shape[0] // batch_size,

          validation_data = (x_val, y_val))





score = model.evaluate(x_val, y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Plotting our loss chart

history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
# Plotting our accuracy charts

history_dict = history.history



acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')

line2 = plt.plot(epochs, acc_values, label='Training Accuracy')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Accuracy')

plt.grid(True)

plt.legend()

plt.show()
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("2.csv",index=False)