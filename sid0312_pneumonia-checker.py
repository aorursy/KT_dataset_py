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
import matplotlib.pyplot as plt

%matplotlib inline
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation

from keras.preprocessing.image import ImageDataGenerator
input_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'

import cv2

def preprocessing(dimensions, batch_size):

    

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    training_generator = train_datagen.flow_from_directory(directory=input_path+'train', target_size=(dimensions, dimensions), batch_size=batch_size, class_mode='binary', shuffle=True)

    test_generator = test_datagen.flow_from_directory(directory=input_path+'test', target_size=(dimensions, dimensions), batch_size=batch_size, class_mode='binary',shuffle=True)

    val_generator = val_datagen.flow_from_directory(directory=input_path+'val', target_size=(dimensions, dimensions), batch_size=batch_size, class_mode='binary',shuffle=True)

    

    test_data = []

    test_labels = []

    

    

    for condition in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(input_path + 'test' + condition)):

            img = plt.imread(input_path+'test'+condition+img)

            img = cv2.resize(img, (dimensions, dimensions))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if condition=='/NORMAL/':

                label = 0

            elif condition=='/PNEUMONIA/':

                label = 1

            test_data.append(img)

            test_labels.append(label)

        

    test_data = np.array(test_data)

    test_labels = np.array(test_labels)



    val_data = []

    val_labels = []

    

    

    for condition in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(input_path + 'test' + condition)):

            img = plt.imread(input_path+'test'+condition+img)

            img = cv2.resize(img, (dimensions, dimensions))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if condition=='/NORMAL/':

                label = 0

            elif condition=='/PNEUMONIA/':

                label = 1

            val_data.append(img)

            val_labels.append(label)

    

    val_data = np.array(test_data)

    val_labels = np.array(test_labels)

    

    

    

    return training_generator,test_generator,val_generator,test_data, test_labels
dimensions= 150

batch_size = 32



train_generator,test_generator,val_generator,test_data,test_labels = preprocessing(dimensions, batch_size)
test_data.shape
model = Sequential()

model.add(Conv2D(64,(3,3),padding="same",input_shape=(150,150,3)))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Conv2D(64,(3,3),padding="same"))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))





model.add(Conv2D(128,(3,3),padding="same"))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Conv2D(128,(3,3),padding="same"))

model.add(MaxPooling2D(2,2))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Conv2D(256,(3,3),padding="same"))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Conv2D(256,(3,3),padding="same"))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dense(256,activation='relu'))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()
from keras.callbacks.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)

learningrate_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
from keras.optimizers import Adam

opt=Adam()

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=int(train_generator.samples/batch_size), epochs=20,validation_data=test_generator,validation_steps=int(test_generator.samples/batch_size), callbacks=[checkpoint, learningrate_reduce])

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from sklearn.metrics import accuracy_score,confusion_matrix

predictions=model.predict(test_data)
acc = accuracy_score(test_labels, np.round(predictions))*100

cm = confusion_matrix(test_labels, np.round(predictions))

tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))

print('Precision: {}%'.format(precision))

print('Recall: {}%'.format(recall))

print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('Training acc: {}%'.format(np.round((history.history['accuracy'][-1])*100, 2)))