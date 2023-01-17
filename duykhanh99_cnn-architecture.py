# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
Y_train = train["label"]

X_train = train.drop(labels=['label'], axis = 1)

X_train = X_train/255

X_test = test/255

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes=10)

# GLOBAL VARIABLES

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']

nets = 3

model = [0]*nets



for j in range(3):

    model[j] = Sequential()

    model[j].add(Conv2D(24, kernel_size=5, padding='same', activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    if j>0:

        model[j].add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))

        model[j].add(MaxPool2D())

    if j>1:

        model[j].add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

        model[j].add(MaxPool2D(padding='same'))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dropout(0.2))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    
# CREATE VALIDATION SET

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)

# TRAIN NETWORKS

history = [0] * nets

names = ["(C-P)x1","(C-P)x2","(C-P)x3"]

epochs = 50

for j in range(nets):

    history[j] = model[j].fit(X_train2,Y_train2, batch_size=128, epochs = epochs, 

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
plt.figure(figsize=(15,5))

for i in range(nets):

    plt.plot(history[i].history['val_acc'], linestyle=styles[i])

plt.title('model accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(names, loc='upper left')

axes = plt.gca()

axes.set_ylim([0.98,1])

plt.show()
nets = 6

model = [0]*nets

for j in range(6):

    model[j] = Sequential()

    model[j].add(Conv2D(j*8+8, kernel_size=5, activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Conv2D(j*16+16, kernel_size=5, activation='relu'))

    model[j].add(MaxPool2D())

    model[j].add(Flatten())

    model[j].add(Dropout(0.2))

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
# CREATE VALIDATION SET

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)

# TRAIN NETWORKS

history = [0] * nets

names = ["8 maps","16 maps","24 maps","32 maps","48 maps","64 maps"]

epochs = 50

for j in range(nets):

    history[j] = model[j].fit(X_train2,Y_train2, batch_size=128, epochs = epochs, 

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# PLOT ACCURACIES

plt.figure(figsize=(15,5))

for i in range(nets):

    plt.plot(history[i].history['val_acc'],linestyle=styles[i])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(names, loc='upper left')

axes = plt.gca()

axes.set_ylim([0.98,1])

plt.show()
nets = 5

model = [0] *nets



j=0

model[j] = Sequential()

model[j].add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model[j].add(MaxPool2D())

model[j].add(Dropout(0.2))

model[j].add(Conv2D(64,kernel_size=5,activation='relu'))

model[j].add(MaxPool2D())

model[j].add(Dropout(0.2))

model[j].add(Flatten())

model[j].add(Dense(128, activation='relu'))

model[j].add(Dropout(0.5))

model[j].add(Dense(10, activation='softmax'))

model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



j=1

model[j] = Sequential()

model[j].add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model[j].add(Conv2D(32,kernel_size=3,activation='relu'))

model[j].add(MaxPool2D())

model[j].add(Dropout(0.2))

model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(MaxPool2D())

model[j].add(Dropout(0.2))

model[j].add(Flatten())

model[j].add(Dense(128, activation='relu'))

model[j].add(Dropout(0.5))

model[j].add(Dense(10, activation='softmax'))

model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



j=2

model[j] = Sequential()

model[j].add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model[j].add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(Dropout(0.4))

model[j].add(Conv2D(64,kernel_size=5,activation='relu'))

model[j].add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(Dropout(0.4))

model[j].add(Flatten())

model[j].add(Dense(128, activation='relu'))

model[j].add(Dropout(0.4))

model[j].add(Dense(10, activation='softmax'))

model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



j=3

model[j] = Sequential()

model[j].add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model[j].add(BatchNormalization())

model[j].add(Conv2D(32,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Dropout(0.4))

model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Dropout(0.4))

model[j].add(Flatten())

model[j].add(Dense(128, activation='relu'))

model[j].add(Dropout(0.4))

model[j].add(Dense(10, activation='softmax'))

model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
j=4

model[j] = Sequential()



model[j].add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model[j].add(BatchNormalization())

model[j].add(Conv2D(32,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Dropout(0.2))



model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Dropout(0.4))



model[j].add(Flatten())

model[j].add(Dense(128, activation='relu'))

model[j].add(BatchNormalization())

model[j].add(Dropout(0.4))

model[j].add(Dense(10, activation='softmax'))



model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# CREATE VALIDATION SET

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.2)

# TRAIN NETWORKS 1,2,3,4

history = [0] * nets

names = ["basic","32C3-32C3","32C5S2","both+BN","both+BN+DA"]

epochs = 50

for j in range(nets-1):

    history[j] = model[j].fit(X_train2,Y_train2, batch_size=64, epochs = epochs,  

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))

    

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.1,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

# TRAIN NETWORK 5

j = nets-1

history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=128), 

    epochs = epochs, steps_per_epoch = X_train2.shape[0]//128,

    validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

    names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# PLOT ACCURACIES

plt.figure(figsize=(15,5))

for i in range(nets):

    plt.plot(history[i].history['val_acc'],linestyle=styles[i])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(names, loc='upper left')

axes = plt.gca()

axes.set_ylim([0.98,1])

plt.show()
# TRAIN OUR BEST NET MORE

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+epochs))

model[4].fit_generator(datagen.flow(X_train,Y_train, batch_size=64), epochs = 25, 

    steps_per_epoch = X_train.shape[0]//64, callbacks=[annealer], verbose=0)



# SUBMIT TO KAGGLE

results = model[4].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST-CNN.csv",index=False)
# BUILD CONVOLUTIONAL NEURAL NETWORKS

from keras.optimizers import RMSprop

nets = 15

model = [0] *nets

for j in range(nets):

    model[j] = Sequential()



    model[j].add(Conv2D(32, kernel_size = 5, activation='relu', input_shape = (28, 28, 1)))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 5, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 3, strides=2, padding='same', activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.2))



    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 3, strides=2, padding='same', activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.2))



    model[j].add(Conv2D(128, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Flatten())

    model[j].add(Dropout(0.4))

    model[j].add(Dense(10, activation='softmax'))



    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

    model[j].compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

# DECREASE LEARNING RATE EACH EPOCH

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN NETWORKS

history = [0] * nets

epochs = 45

for j in range(nets):

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),

        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# ENSEMBLE PREDICTIONS AND SUBMIT

results = np.zeros( (X_test.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)
plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("predict=%d" % results[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()