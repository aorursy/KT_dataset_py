# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import cv2 # IMAGE PROCESSING - OPENCV

from glob import glob # FILE OPERATIONS

import itertools



# KERAS AND SKLEARN MODULES

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import BatchNormalization

from keras.utils import to_categorical



from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
image_dir = '../input/train/*/*.png'

images = glob(image_dir)

training_images = []

training_labels = []

num = len(images)

count = 1



#READING IMAGES AND RESIZING THEM

for i in images:

    print(str(count)+'/'+str(num),end='\r')

    training_images.append(cv2.resize(cv2.imread(i),(128,128)))

    training_labels.append(i.split('/')[-2])

    count=count+1

training_images = np.asarray(training_images)

training_labels = pd.DataFrame(training_labels)
training_labels.rename(columns={0: 'sappling'}, inplace=True)

training_labels.sappling.value_counts()

plt.figure(figsize=(20,10))

sns.barplot(x=training_labels.sappling.unique(), y=training_labels.sappling.value_counts())


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

fig.suptitle('Sharing x per column, y per row')

ax1.imshow(training_images[3])

ax2.imshow(training_images[4])

ax3.imshow(training_images[5])

ax4.imshow(training_images[6])

ax5.imshow(training_images[7])

ax6.imshow(training_images[8])

new_train = []

sets = []; getEx = True

for i in training_images:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    #GREEN PARAMETERS

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    new = np.zeros_like(i,np.uint8)

    new[boolean] = i[boolean]

    new_train.append(new)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL

        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED

        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED

        plt.subplot(2,3,4);plt.imshow(mask) # MASKED

        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED

        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE

        plt.show()

        getEx = False

new_train = np.asarray(new_train)



# CLEANED IMAGES

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(new_train[i])
def createModel():

    model = Sequential()

    

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.1))



    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.1))



    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.1))



    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.1))

    



    model.add(Flatten())



    model.add(Dense(4096, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(4096, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(12, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    model.summary()

    

    return model
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()





# 2/3. FIT AND TRANSFORM

# use df.apply() to apply le.fit_transform to all columns

training_labels = training_labels.apply(le.fit_transform)

print(training_labels.head(20))

print(type(training_labels))



training_labels = np.array(training_labels)
training_labels = to_categorical(training_labels)
sappling_model = createModel()

history = sappling_model.fit(training_images, training_labels, batch_size=32, epochs=20, verbose=1)
def save_model(model):

    sappling_model_json = model.to_json()

    with open("sappling_model.json", "w") as json_file:

        json_file.write(sappling_model_json)

    # serialize weights to HDF5

    model.save_weights("sappling_model.h5")

    print("Saved model to disk")
save_model(sappling_model)