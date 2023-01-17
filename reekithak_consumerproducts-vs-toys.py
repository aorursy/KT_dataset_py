# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

'''import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('../input')
from tensorflow.keras.layers import Input , Dense, Flatten,Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.applications.resnet50 import ResNet50



from tensorflow.keras.applications.resnet50 import preprocess_input



from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.models import Sequential

import numpy as np

from glob import glob

import matplotlib.pyplot as plt

%matplotlib inline





from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input



import tensorflow as tf 

from keras.models import load_model

from keras import optimizers, losses, activations, models
from keras.models import Sequential

from keras.models import Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import optimizers, losses, activations, models

from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
img_size = [150,150]

train_path = '/kaggle/input/consumer-products-vs-toys/Data/train'

test_path = '/kaggle/input/consumer-products-vs-toys/Data/test'

#vgg = VGG16(input_shape=img_size+[3] , weights='imagenet',include_top= False)

#vgg.layers

new_model = applications.InceptionV3(weights='imagenet', 

                                include_top=False, 

                                input_shape=(150, 150,3))
'''new_model.summary()'''
for layer in new_model.layers:

    layer.trainable = False
'''new_model.summary()'''
#Adding ending layers

#x1 = Dropout(0.3)(vgg.output)

x = Flatten()(new_model.output)#Final layer

#x2 = Dropout(0.2)(x)

#x3 = Flatten()(x2)#Final layer

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

prediction = Dense(1,activation='sigmoid')(x)

model = Model(inputs=new_model.input,outputs=prediction)
'''model.summary()'''
import tensorflow as tf

from tensorflow import keras

import numpy as np
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
from keras.preprocessing.image import ImageDataGenerator

training_set = train_datagen.flow_from_directory('/kaggle/input/consumer-products-vs-toys/Data/train',

                                                target_size=(150,150),

                                                batch_size=32,

                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('/kaggle/input/consumer-products-vs-toys/Data/test',

                                                target_size=(150,150),

                                                batch_size=32,

                                                class_mode='binary')

import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self,epoch,logs={}):

            if(logs.get('accuracy')>0.955 and logs.get('val_accuracy')>0.915):

                print("\nReached n% accuracy so cancelling training!")

                self.model.stop_training = True
callbacks=MyCallback()
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)



early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')



fit_  = model.fit(training_set,

                           validation_data=test_set,

                           epochs=25,

                           steps_per_epoch=len(training_set),

                           validation_steps=len(test_set),verbose=1,

                  callbacks = [callbacks,checkpoint,early])
fit_ = model.fit(training_set,

                           validation_data=test_set,

                           epochs=300,

                           steps_per_epoch=len(training_set),

                           validation_steps=len(test_set),verbose=1,

                  callbacks = [callbacks,checkpoint,early],initial_epoch=25)
'''model.save("final_model.h5")'''
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(fit_.history['accuracy'])

plt.plot(fit_.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','val'],loc='upper left')

plt.show()

plt.plot(fit_.history['loss'])

plt.plot(fit_.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('epoch')

plt.ylabel('Loss')

plt.legend(['train','val'],loc='upper left')

plt.show()

model.save('inception_new.h5')