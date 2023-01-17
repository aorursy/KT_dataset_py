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
import time

import pandas as pd

import os,shutil,math

import numpy as np

import matplotlib.pyplot as plt



from sklearn.utils import shuffle

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,roc_curve,auc



from PIL import Image

from PIL import ImageDraw

from glob import glob





from IPython.display import SVG



from keras.utils.vis_utils import model_to_dot

from keras.applications.vgg19 import VGG19,preprocess_input

from keras.applications.xception import Xception

from keras.applications.nasnet import NASNetMobile

from keras.models import Sequential,Input,Model

from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,BatchNormalization,Activation

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, ModelCheckpoint

import numpy as np

import pandas as pd

import cv2

import os

from glob import glob

import matplotlib.pyplot as plt

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input, decode_predictions
train_data_dir = '../input/flowers/flowers'

img_width, img_height = 299, 299 

batch_size = 32

# Bilder haben unterschiedliche Dimensionen.


image_datagen = ImageDataGenerator(

    rescale=1./255, 

    vertical_flip = True,

    horizontal_flip = True,

    rotation_range=20,

    shear_range=0.05,

    zoom_range=0.2,

    width_shift_range=0.1,

    height_shift_range=0.1,

    validation_split=0.2

    #channel_shift_range=0.1

)

train_gen = image_datagen.flow_from_directory(

        train_data_dir, 

        target_size=(img_height, img_width), 

        batch_size=batch_size, 

        class_mode="categorical", 

        subset="training")

valid_gen = image_datagen.flow_from_directory(

        train_data_dir, 

        target_size=(img_height, img_width), 

        batch_size=batch_size, 

        class_mode="categorical", 

        subset="validation")
# In the summary above of our base model, trainable params is 2,565



# Callbacks

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=10,

    verbose=1,

    mode='auto' 

)

csvlogger = CSVLogger( 

    filename= "training_csv.log",

    separator = ",",

    append = False

)

reduce = ReduceLROnPlateau( 

    monitor='val_loss',

    factor=0.1, 

    patience=3, #kac epoch iyilesmezse learning rate düssün

    verbose=1, 

    mode='auto',

)    



# Hyperparameters

second_dense_512 = [0, 1]

dropout = [0, 1]



base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

print('Loaded model!')


# Freeze the layers in base_model 

for layer in base_model.layers:

    layer.trainable = False 

        

for dense2 in second_dense_512:

    for drop in dropout:

        

        NAME = "flowers-inception-dense{}-drop{}-{}".format(dense2, drop, int(time.time()))

        print(NAME)

        logdir = "logs/flowers-inception/{}/".format(NAME)

        

        # Callbacks

        checkpoint = ModelCheckpoint(

            '{}base.model'.format(logdir),

            monitor='val_loss',

            mode='min',

            save_weights_only=True,

            save_best_only = True,

            verbose = 1)

        tensorboard = TensorBoard(

            log_dir = logdir,

            histogram_freq=0,

            batch_size=batch_size,

            write_graph=True,

            write_grads=True,

            write_images=False,

        )



        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        x = Dense(1024)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        if drop == 1 : x = Dropout(0.3)(x)

        if dense2 == 1 : 

            x = Dense(512)(x)

            x = BatchNormalization()(x)

            x = Activation("relu")(x)

            if drop == 1 : x = Dropout(0.3)(x)

        

        predictions = Dense(5, activation='softmax')(x)

        

        model = Model(base_model.input, predictions)

        

        

        model.compile(loss='categorical_crossentropy',

                      optimizer='Adam',

                      metrics=['accuracy'])

        

        # Take a look at layers of model

        '''

        pd.set_option('max_colwidth', -1)

        layers = [(layer, layer.name, layer.trainable) for layer in model.layers]

        print(pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) )

        '''

    

        history = model.fit_generator(

            train_gen,

            steps_per_epoch = train_gen.n // train_gen.batch_size, #normalde len(X_train) / batch_size,

            epochs= 100,

            validation_data = valid_gen,

            validation_steps=valid_gen.n // valid_gen.batch_size, # normalde len(X_valid) / batch_size,

            verbose=1,

            callbacks=[checkpoint,tensorboard,csvlogger,reduce,earlystop])

       

        



'''

model.load_weights(best_model_finetuned_path)  

   

(eval_loss, eval_accuracy) = model.evaluate(  

     X_test, y_test, batch_size=batch_size, verbose=1)



print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  

print("Loss: {}".format(eval_loss)) 

'''
