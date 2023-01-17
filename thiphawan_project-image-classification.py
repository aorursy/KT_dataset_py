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
# Importing the Keras libraries and packages



from keras.models import Sequential   

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras import regularizers

from keras import optimizers as Optimizer

from keras.layers import Dropout

from keras.models import load_model

import keras.backend as K

import numpy as np

from sklearn.metrics import confusion_matrix

from matplotlib import image

import matplotlib.pyplot as plt

import cv2

from sklearn.utils import shuffle

from random import randint



test_dir="../input/intel-image-classification/seg_test/seg_test"

train_dir="../input/intel-image-classification/seg_train/seg_train"
# Initializing the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(128, (5, 5), activation='relu',input_shape=(150, 150, 3),padding='same',kernel_regularizer=regularizers.l2(0.0025)))



# Step 2 - Maxpooling

classifier.add(MaxPooling2D((3, 3)))

classifier.add(Dropout(0.5))



classifier.add(Conv2D(128, (5, 5), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0025)))

classifier.add(MaxPooling2D((5, 5)))

classifier.add(Dropout(0.3))

classifier.add(Conv2D(256, (5, 5), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0025)))

classifier.add(MaxPooling2D((2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())







# Step 4 - Full Connection to the ANN

#classifier.add(Dense(units = 128, activation = 'relu'))

#classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.002)))

classifier.add(Dropout(0.5))

classifier.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.002)))

classifier.add(Dense(6, activation='softmax'))





# Step 5 - Compiling

classifier.compile(loss='categorical_crossentropy', optimizer=Optimizer.Adam(lr=0.0008),metrics=['accuracy'])
classifier.summary()
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)

#train_generator

training_set = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=100,class_mode='categorical')

#validation_generator

test_set = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=100,class_mode='categorical')
trained = classifier.fit_generator(training_set,steps_per_epoch=150,epochs=6,validation_data=test_set,validation_steps=60)
fig = plt.figure(figsize=(10,5))

plt.subplot(221)

plt.plot(trained.history['accuracy'], label = "Train")

plt.plot(trained.history['val_accuracy'], label = "Test")

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.legend()



plt.subplot(222)

plt.plot(trained.history['loss'], label = "Train")

plt.plot(trained.history['val_loss'], label = "Test")

plt.title("Model Loss")

plt.ylabel("Loss")

plt.xlabel("Epochs")





plt.legend()

plt.show()
def data_making_for_prediction(directory):

    data=[]

    for img in os.listdir(directory):    

        path_img=os.path.join(directory,img)

        img_data=cv2.resize(cv2.imread(path_img),(150,150))

        data.append((img_data))



    #shuffle data 

    shuffle(data)

    return data
pred_Images=data_making_for_prediction('../input/intel-image-classification/seg_pred/seg_pred')

pred_Images = np.array(pred_Images)
def get_classlabel(class_code):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    return labels[class_code]
f,ax = plt.subplots(2,2) 

f.subplots_adjust(0,0,3,3)

for i in range(0,2,1):

    for j in range(0,2,1):

        rnd_number = randint(0,len(pred_Images))

        ax[i,j].imshow(pred_Images[rnd_number])

        ax[i,j].set_title(get_classlabel(classifier.predict_classes(np.array(pred_Images[rnd_number]).reshape(-1,150,150,3))[0]))

        ax[i,j].axis('off')