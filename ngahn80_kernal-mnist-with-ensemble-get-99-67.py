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
# import some library to use

from tensorflow import keras   # keras for DL

import matplotlib.pyplot as plt  # for data visualization

import sklearn

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, BatchNormalization, Flatten

from keras.preprocessing.image import ImageDataGenerator



# read data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

Y = train['label']

X = train.drop(axis =1, columns = 'label')

# reshape for visualize data

X = X.values.reshape(-1,28,28)

Y = Y.values

test = test.values.reshape(-1,28,28)



x_train,x_val,y_train, y_val = train_test_split(X,Y,test_size = 0.2, random_state = 8)

x_train = x_train / 255

x_val = x_val /255

y_train = to_categorical(y_train,num_classes=10)

y_val = to_categorical(y_val,num_classes=10)

test = test /255

x_train = x_train.reshape(-1,28,28,1)

x_val = x_val.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
epoch = 50

nets = 10

batch = 64

model = [0]*nets

for i in range(nets):

    model[i] = Sequential()

    model[i].add(Conv2D(filters=64, kernel_size=3,activation='relu',padding='same',input_shape = (28,28,1)))        

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=64, kernel_size=3,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=128, kernel_size=3,strides=2,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Dropout(0.4))

    

    model[i].add(Conv2D(filters=128, kernel_size=3,activation='relu',padding='same',input_shape = (28,28,1)))        

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=256, kernel_size=3,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=256, kernel_size=3,strides=2,activation='relu',padding='same'))

    model[i].add(BatchNormalization())

    model[i].add(Dropout(0.4))

    

    model[i].add(Flatten())

    model[i].add(Dense(units=128,activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Dropout(0.4))

    model[i].add(Dense(units=10,activation='softmax'))



    model[i].compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# model[0].summary()

    


datagen = ImageDataGenerator(

                        rotation_range = 10,

                        zoom_range =0.1,

                        width_shift_range = 0.1,

                        height_shift_range = 0.1

                        )

history = [0]*nets

for i in range(nets):

    filepath="weight.net{}.best.hdf5".format(i)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_acc',verbose=0,save_best_only=True,mode='max')

    callback_lists = [checkpoint]

    history[i] = model[i].fit_generator(datagen.flow(x_train,y_train,batch_size = batch),steps_per_epoch = x_train.shape[0]//batch, epochs = epoch,validation_data = (x_val,y_val),verbose=0,callbacks = callback_lists)                   

#     history = model[i].fit(x_train,y_train, epochs = epoch,validation_data = (x_val,y_val),verbose=1)                   

    print("CNN-{}: has {} train_accuracy at {} epoch and {} val_accuracy at {} epoch".format(i+1,max(history[i].history['acc']),np.argmax(history[i].history['acc']),max(history[i].history['val_acc']),np.argmax(history[i].history['val_acc'])))

    model[i].load_weights(filepath)

def display_accuracy(history,nets):

    #Display the acc of result

    

    plt.figure(figsize=(15,5))

    for i in range(nets):

        plt.plot(history[i].history['val_acc'])

    plt.title('model accuracy')

    plt.xlabel("Epochs")

    plt.ylabel('Accuracy')

    plt.legend(range(nets))

    plt.ylim((0.97,1))

    plt.show()

display_accuracy(history,nets)
#predict

predictions = np.zeros((test.shape[0],10))

for i in range(nets):

    predictions = predictions+ model[i].predict(test)

predictions = np.argmax(predictions,axis=1)

submissions = pd.DataFrame({'ImageID':list(range(1,len(predictions)+1)),'Label':predictions})

# keras.utils.plot_model(model_cnn,to_file='model.png')

# model_cnn.save("model.h5")



#submission

submissions.to_csv('submission.csv',index = False, header = True)