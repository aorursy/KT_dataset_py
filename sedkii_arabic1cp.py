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
import numpy as np # linear algebra
import xml.etree.ElementTree as ET
#lab=np.array([])
lab=[]
#tree = ET.ElementTree(file='../input/ba3adads/PetitCorpus_Racine0_tab3oudou.xml');
#tree = ET.ElementTree(file='../input/teb3odouds/CorpusApprentissage_Racine0.xml');
tree = ET.ElementTree(file='../input/ba3adads22/CorpusApprentissage_Racine0.xml');
root=tree.getroot();
#print(root)
children = root.getchildren();
print(children)
subchld = children[0].getchildren();
for sbs in subchld:
    #print(sbs.get('Dist0'))
    lab.append(sbs.get('Dist0'))
 #   np.append(lab,sbs.get('Dist0'))
 #   ET.dump(chld)
##############################
print(np.shape(lab))
#print(np.shape(Y_train))
print(lab[:])
#Determination de x_train et y_train
import numpy as np # linear algebra
import os
import glob as gb
import cv2
s1=64
s2=64
x_train = []
y_train = []
#trainpath='../input/ba3adads/BA3ADA/'
#trainpath='../input/teb3odouds/teb3odou/teb3odou/'
#trainpath='../input/ba3adads2/BA3ADA/'
trainpath='../input/ba3adads22/BA3ADA2/BA3ADA2/'
#for folder in  os.listdir('../input/ba3adads/BA3ADA') : 
#for folder in  os.listdir('../input/teb3odouds/teb3odou/teb3odou/') : 
#for folder in  os.listdir('../input/ba3adads2/BA3ADA') : 
for folder in  os.listdir('../input/ba3adads22/BA3ADA2/BA3ADA2/') : 
    files = gb.glob(pathname= str( trainpath + folder +'/*.png'))
    for file in files: 
        image = cv2.imread(file)/ 255.0
        image_array = cv2.resize(image , (s1,s2))
        x_train.append(list(image_array))
        #y_train.append(int(lab[int(folder)]))
        y_train.append(int(folder))
##################################################
print(np.shape(x_train))
print(np.shape(y_train))
#print(np.shape(y_train))
#print(type(x_train))
#x_train = np.array(x_train) 
#print(np.shape(x_train))
#print(type(x_train))
import numpy as np # linear algebra
y_train = np.array(y_train) 
#print(type(y_train))
#print(np.shape(y_train))
y_train
y_train = y_train.astype('int32')
print(np.unique(y_train))
print(np.shape(x_train))
#print(type(x_train))
x_train = np.array(x_train) 
print(np.shape(y_train))
print(type(x_train))
print(type(y_train))
#######################
print(type(y_train[7]))
print(y_train[0:20])
c=y_train
import matplotlib.pyplot as plt
# Some examples
g = plt.imshow(x_train[5][:,:,:])
print(y_train[0])
print(x_train[0])
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
batch_size = 64
num_classes = 6
epochs = 20
input_shape = (64, 64, 3)
y_train
# convert class vectors to binary class matrices One Hot Encoding
#y_train = y_train -2
y_train = y_train -1
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
model.summary()
datagen.fit(x_train)
epochs = 50
# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size)
