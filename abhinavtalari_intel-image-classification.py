# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import os

# import tensorflow as tf

# import numpy as np

# import glob

# import math

# from keras.preprocessing import image 



# def load_dataset():

    

#     train_set_x = []

#     train_set_y = []

#     test_set_x = []

#     test_set_y = []

#     classes = []

  

#     types = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

#     for x in types:



#         # For loading training set

#         train_files = glob.glob("/kaggle/input/intel-image-classification/seg_train/seg_train/" + str(x) + "/*.jpg")       

#         for pic in train_files:



#             pic = image.load_img(pic, target_size=(150,150))

#             train_image = image.img_to_array(pic) 



#             train_set_x.append(train_image)

#             train_set_y.append(int(types[x]))



#         # For loading test set

#         test_files = glob.glob("/kaggle/input/intel-image-classification/seg_test/seg_test/" + str(x) + "/*.jpg")

#         for pic in test_files:

            

#             pic = image.load_img(pic, target_size=(224, 224))

            

#             test_image = image.img_to_array(pic)



#             test_set_x.append(test_image)

#             test_set_y.append(int(types[x]))

#         # For loading classes

#         classes.append(int(types[x]))



#     classes = np.array(classes)

#     classes=classes.reshape(1,classes.shape[0])

#     print(classes)

#     print(classes.shape)

#     train_set_y = np.array(train_set_y)

#     train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

#     train_set_x = np.array(train_set_x)

#     print(train_set_x.shape)

#     print(train_set_y.shape)



#     test_set_y = np.array(test_set_y)

#     test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

#     test_set_x = np.array(test_set_x)

#     print(test_set_x.shape)

#     print(test_set_y.shape)



#     return train_set_x, train_set_y, test_set_x, test_set_y, classes
# from sklearn.model_selection import train_test_split

# from keras.utils.np_utils import to_categorical

# from keras.models import Sequential

# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization

# from keras.optimizers import RMSprop

# from keras.preprocessing.image import ImageDataGenerator

# from keras.callbacks import ReduceLROnPlateau, CSVLogger



# train_set_x, train_set_y, test_set_x, test_set_y, classes=load_dataset()

# train_set_y = to_categorical(train_set_y, num_classes=10)

# print(train_set_y)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.,horizontal_flip=True,shear_range=0.2,  

    zoom_range=0.2, validation_split=0.1)



train_generator=train_datagen.flow_from_directory(

    '/kaggle/input/intel-image-classification/seg_train/seg_train/',

      target_size=(150,150),

      batch_size=64,

      class_mode='sparse',

      seed=2209,

      subset='training'   

)



validation_generator=train_datagen.flow_from_directory(

      '/kaggle/input/intel-image-classification/seg_train/seg_train/',

      target_size=(150,150),

      batch_size=64,

      class_mode='sparse',

      seed=2209,

      subset='validation'   

)



test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_directory(

    '/kaggle/input/intel-image-classification/seg_test/seg_test/', 

    target_size=(150,150),

    batch_size=32,

    class_mode='sparse',

    seed=2209

) 





import tensorflow as tf

from keras import regularizers



model=tf.keras.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),

    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),



    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),



    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),





    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    

    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),

    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),



    

    tf.keras.layers.Dense(32,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),



    

    tf.keras.layers.Dense(16,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Dense(6,activation='softmax')

    





    

])

model.summary()





# from keras.layers import *

# from keras.models import Sequential

# from keras.models import Model

# from keras.layers import Dense

# from keras.layers import Dropout

# from keras.layers import BatchNormalization

# import numpy as np 

# from keras.applications.resnet50 import ResNet50

# import keras





# parent_model=ResNet50(include_top=False, weights= 'imagenet',input_shape=(150,150,3),pooling='avg')

# parent_model.trainable=False



# x = parent_model.output

# x = Dense(512,activation='relu')(x) #dense layer 3

# # x = Dropout(0.5)(x)

# x = Dense(256,activation='relu')(x) #dense layer 4

# x = Dense(128,activation='relu')(x) #dense layer 5

# x = Dense(64,activation='relu')(x) #dense layer 6

# x = Dense(6,activation='softmax')(x) #final layer with softmax activation



# child_model=Model(parent_model.input,x)

# child_model.compile(optimizer =keras.optimizers.SGD(lr=0.0001), 

#               loss = 'sparse_categorical_crossentropy', 

#               metrics = ['accuracy'])



# child_model.summary()
import tensorflow as tf

model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')

history = model.fit_generator(

            train_generator,

            steps_per_epoch=int(12632/64),

            epochs=100,

            validation_data=validation_generator,

            validation_steps=int(1402/64),

            verbose=1)
accuracy=model.evaluate_generator(test_generator, steps=len(test_generator)//32,verbose=2)

print('Accuracy of the model on the test set: ',accuracy[1])




#Evaluating Accuracy and Loss of the model

%matplotlib inline

acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) #No. of epochs



#Plot training and validation accuracy per epoch

import matplotlib.pyplot as plt

plt.plot(epochs,acc,'b',label='Training Accuracy')

plt.plot(epochs,val_acc,'r',label='Validation Accuracy')

plt.legend()

plt.figure()



#Plot training and validation loss per epoch

plt.plot(epochs,loss,'b',label='Training Loss')

plt.plot(epochs,val_loss,'r',label='Validation Loss')

plt.legend()

plt.show()


