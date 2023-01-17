# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.preprocessing.image import ImageDataGenerator#Generate batches of tensor image data with real-time data augmentation.
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('../input/face-mask-12k-images-dataset/Face Mask Dataset/Train',target_size=(127,127),batch_size=32,class_mode='binary')

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_datagen.flow_from_directory('../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation', target_size = (127,127), batch_size = 32, class_mode = 'binary')
from tensorflow.keras.models import Sequential#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
from tensorflow.keras.layers import Dropout,Activation, Conv2D,MaxPooling2D, Flatten, MaxPool2D, Dense, BatchNormalization#1)Dense layer is the regular deeply connected neural network layer.
#Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly,2)Flattens the input. Does not affect the batch size.
#Max pooling operation for 2D spatial data.
#(conv+relu)*2+maxpol+(conv+relu)*2+maxpol+faltt+(dens+relu)*2+(dens+sigmoid)

cnn = Sequential()

cnn.add(Conv2D(32, activation = 'relu', kernel_size = 3, input_shape = [127,127,3]))
cnn.add(BatchNormalization())

cnn.add(Conv2D(32, activation = 'relu', kernel_size = 3))
cnn.add(BatchNormalization())


cnn.add(MaxPool2D(pool_size = (2,2)))

cnn.add(Conv2D(64, activation = 'relu', kernel_size = 3))
cnn.add(BatchNormalization())

cnn.add(Conv2D(64, activation = 'relu', kernel_size = 3))
cnn.add(BatchNormalization())

cnn.add(MaxPool2D(pool_size = (2,2)))

cnn.add(Flatten())

cnn.add(Dense(64, activation = 'relu'))
cnn.add(Dropout(0.2))

cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dropout(0.2))

cnn.add(Dense(1, activation = 'sigmoid'))

cnn.compile(optimizer='adam', metrics = 'accuracy', loss = 'binary_crossentropy')

def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    #model.summary()
    
    return model
model=my_model()
model.summary()
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(training_set, validation_set, test_size = 0.1, random_state=2)
history = model.fit_generator( datagen.flow( X, y, batch_size = BATCH_SIZE, shuffle = True),
        samples_per_epoch = len(X), nb_epoch = 15, callbacks = callbacks,
        validation_data = ( X, y ), verbose = 1, show_accuracy = True )
history=model.fit(training_set,epochs = 3, verbose=2, validation_data =validation_set)

batch_size=32

history = model.fit(training_set,validation_set, batch_size,epochs=3,verbose=2)#,callbacks = [learning_rate_reduction])
from keras.preprocessing import image#Set of tools for real-time data augmentation on image data.
test_image = image.load_img('../input/face-mask-12k-images-dataset/Face Mask Dataset/Test/WithMask/1163.png', target_size = (127,127,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

answer = cnn.predict_classes(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction = 'mask'
else :
    prediction = 'unmask'
    
print(prediction)
