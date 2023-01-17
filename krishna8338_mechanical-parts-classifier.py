from PIL import Image

from skimage.io import imread

from skimage.transform import rescale, resize

import os

import glob

import string

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras import applications

import keras

from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix

import seaborn as sn

import shutil

import random

import cv2
#Creation of a CNN . Sequential Model



model = Sequential()

#input_shape matches our input image

model.add(Conv2D(64, (3,3), input_shape=(224, 224, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(4)) #data of four types

model.add(Activation('softmax'))



model.compile(loss= keras.losses.categorical_crossentropy, 

              optimizer= Adam(),metrics=['accuracy'])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        

        rotation_range=90, 

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=True,  

        vertical_flip=True,

        rescale = 1./255,

        validation_split = 0.2

        )  



test_datagen = ImageDataGenerator(rescale = 1./255 )
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
train_path = '/kaggle/input/mechanical-parts-data/dataset/training'

test_path = '/kaggle/input/mechanical-parts-data/dataset/testing'
# # Make sure you provide the same target size as initialied for the image size

training_set = train_datagen.flow_from_directory(train_path,

                                            target_size = (224, 224),

                                            batch_size = 32,

                                            class_mode = 'categorical',

                                            subset = 'training',

                                            shuffle = True)
val_set = train_datagen.flow_from_directory(train_path,

                                            target_size = (224, 224),

                                            batch_size = 32,

                                            class_mode = 'categorical',

                                            subset = 'validation',

                                            shuffle = True)
test_set = test_datagen.flow_from_directory(test_path,

                                            target_size = (224, 224),

                                            batch_size = 32,

                                            class_mode = 'categorical',

                                            shuffle = True)
history = model.fit(

  training_set,

  validation_data= test_set,

  epochs= 3,

  shuffle = True,

  steps_per_epoch=len(training_set),

  validation_steps=len(test_set)

  

)
model.evaluate(test_set)
model.save('mech2.h5')
#history.history
loss_train = history.history['loss']

loss_val = history.history['val_loss']

#epochs = range(1,no_epochs+1)

plt.plot( loss_train, 'g', label='Training loss')

plt.plot( loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()

plt.savefig('train-val loss')
loss_train = history.history['accuracy']

loss_val = history.history['val_accuracy']

#epochs = range(1,no_epochs+1)

plt.plot( loss_train, 'g', label='Training Accuracy')

plt.plot( loss_val, 'b', label='validation Accuracy')

plt.title('Training and Validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

plt.savefig('train-val-accuracy')
#Load the model 



from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

import numpy as np



#Loading our Model

model = load_model('./mech2.h5')
import os 

import random 

import cv2

import matplotlib.pyplot as plt

import glob

%matplotlib inline 



test_path = '/kaggle/input/mechanical-parts-data/dataset/testing'



list = glob.glob(test_path +'/*')



path =  random.choice(list) 

pic = random.choice(glob.glob(path + '/*'))



pict =  cv2.imread(pic)

plt.imshow(pict)

plt.title(pic)
img=image.load_img(pic,target_size=(224,224))

x=image.img_to_array(img)

x
x=x/255

print(x)
x.shape


x=np.expand_dims(x,axis=0)

x.shape
model.predict(x)
a=np.argmax(model.predict(x))
a
if(a==0):

    plt.imshow(img)

    plt.title('Bolt')

elif(a == 1):

    plt.imshow(img)

    plt.title('Locating Pin')

    

elif(a==2):

    plt.imshow(img)

    plt.title('Nut')



elif(a==3):

    plt.imshow(img)

    plt.title('Washer')