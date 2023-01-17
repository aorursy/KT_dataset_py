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

        pass

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder 

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix,precision_score,recall_score

from keras.callbacks import ReduceLROnPlateau
train_dir = '../input/genderdetectionface/dataset1/train'

test_dir = '../input/genderdetectionface/dataset1/test'

valid_dir = '../input/genderdetectionface/dataset1/valid'
train_folders = os.listdir(train_dir)

train_woman = os.listdir(train_dir+'/'+train_folders[0])

train_woman

data = plt.imread(train_dir+'/'+train_folders[0]+'/'+train_woman[0])

data = cv2.resize(data,(150,150),interpolation = cv2.INTER_AREA)

plt.imshow(data)

plt.show()


def label_images(directory):

    x= []

    y =[] 

    folders = os.listdir(directory)

    for i in range(len(folders)):

        images_list = os.listdir(directory+'/'+folders[i])

        for each in images_list:

            img = cv2.imread(directory+'/'+folders[i]+'/'+each)

            img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)

            x.append(np.array(img))

            y.append(str(folders[i]))

    print(len(x))

    print(len(y))

    return np.array(x),np.array(y)

            

x_train,y_train_labels = label_images(train_dir)

x_test,y_test_labels = label_images(test_dir)

x_valid, y_valid_labels = label_images(valid_dir)

def encoding_target(target_labels):

    le = LabelEncoder()

    

    le.fit(target_labels)

    target_labels = le.transform(target_labels)

    

    target_labels = to_categorical(target_labels)

    return target_labels

y_train = encoding_target(y_train_labels)

y_valid = encoding_target(y_valid_labels)

y_test = encoding_target(y_test_labels)



callback1 = ReduceLROnPlateau(monitor='val_acc',patience=3,factor=0.1,verbose=1)

callback2 =ModelCheckpoint('weights.hdf5',verbose=1,monitor='val_accuracy',save_best_only=True)

callback_list=[callback1,callback2]
model= Sequential()

model.add(Conv2D(56,kernel_size=3,padding='same',input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),activation='relu'))

model.add(MaxPooling2D(2))



model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling2D(2))



model.add(Conv2D(96, kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling2D(2))



model.add(Conv2D(128,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling2D(2))



model.add(Flatten())



model.add(Dense(512,activation='relu'))





model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',metrics =['accuracy'],loss='binary_crossentropy')
datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

datagen.fit(x_train)
history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),

                    steps_per_epoch=len(x_train) / 32, epochs=50,validation_data=(x_valid,y_valid),callbacks=callback_list)
model.load_weights('weights.hdf5')

# print(model.get_weights())

# print(model.history['val_accuracy'])

preds_digits=model.predict_classes(x_test)
plt.plot(history.history['accuracy'],'*-')

plt.plot(history.history['val_accuracy'],'*-')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(history.history['loss'],'*-')

plt.plot(history.history['val_loss'],'*-')

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
preds = model.predict(x_test)

# preds,y_test[:10]
matched_class=[]

mismatched_class=[]

for i in range(len(y_test)):

    if np.argmax(y_test[i],)==np.argmax(preds[i]):

        matched_class.append(i)

    else:

        mismatched_class.append(i)

print(matched_class)

print(mismatched_class)

len(matched_class)
le = LabelEncoder()

le.fit(y_test_labels)

fig, ax = plt.subplots(4,5,figsize=(30,20))

fig.set_size_inches(15,15)



count=270

for i in range(4):

    for j in range(5):

        ax[i,j].imshow(x_test[matched_class[count]])

        ax[i,j].set_title([preds_digits[matched_class[count]]])

        count+=1

        
fig, ax = plt.subplots(4,5,figsize=(30,20))

fig.set_size_inches(15,15)

count=0

for i in range(4):

    for j in range(5):

        ax[i,j].imshow(x_test[mismatched_class[count]])

        ax[i,j].set_title(preds_digits[mismatched_class[count]])

        count+=1


actual=[]

for i in range(len(y_test)):

    actual.append(np.argmax(y_test[i]))

predicted = model.predict_classes(x_test)

print(confusion_matrix(actual,predicted))

precision_score(actual,predicted)