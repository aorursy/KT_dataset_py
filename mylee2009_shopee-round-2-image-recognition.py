# Inspiration from https://github.com/stratospark/food-101-keras
import os

import random



import tensorflow as tf

print(tf.__version__)



from tensorflow.keras import applications

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, AveragePooling2D, Reshape

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.regularizers import l2

from tensorflow.keras.initializers import glorot_uniform



import numpy as np

from sklearn.utils.class_weight import compute_class_weight
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    

x = base_model.output



x = AveragePooling2D(pool_size=(8, 8))(x)

x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Flatten()(x)



predictions = tf.keras.layers.Dense(42, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)



opt = SGD(lr=.01, momentum=.9)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



rows,cols = 299,299



data_path = '../input/shopee-round-2-product-detection-challenge/train/train/' # change this path to link to the food data
# from keras.models import model_from_json

# serialize model to JSON

model_json = model.to_json()

with open("model_xception.json", "w") as json_file:

    json_file.write(model_json)
train_generator = ImageDataGenerator(

                    rescale=1./255,

                    validation_split=0.05

#                     rotation_range=10,

#                     width_shift_range=0.2,

#                     height_shift_range=0.2,

#                     shear_range=0.2,

#                     zoom_range=0.2,

#                     horizontal_flip=True,

#                     fill_mode='reflect'

 )



# val_generator = ImageDataGenerator(

#                     rescale=1./255,

#                     validation_split=0.05)



train_dataset = train_generator.flow_from_directory(

                                                    seed=16,

                                                    batch_size=64,

                                                    directory='../input/shopee-round-2-product-detection-challenge/train/train/',

                                                    shuffle=True,

                                                    subset="training",

                                                    class_mode='categorical',

                                                    target_size=(299, 299))



validation_dataset = train_generator.flow_from_directory(

                                                    seed=16,

                                                    batch_size=64,

                                                    directory='../input/shopee-round-2-product-detection-challenge/train/train/',

                                                    shuffle=True,

                                                    target_size=(299, 299), 

                                                    subset="validation",

                                                    class_mode='categorical')
checkpoint = ModelCheckpoint("model_intermediate_Xception.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only = False,

                             verbose=1)



# earlystop = EarlyStopping(monitor = 'val_loss', 

#                           min_delta = 0, 

#                           patience = 2,

#                           verbose = 1,

#                           restore_best_weights = False)



# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',

#                               factor = 0.5,

#                               patience = 1,

#                               verbose = 1,

#                               min_delta = 0.0003)

def schedule(epoch):

    if epoch <= 2:

        return .008

    elif epoch <= 4:

        return .004

    elif epoch <= 6:

        return .0024

    else:

        return .0004

lr_scheduler = LearningRateScheduler(schedule)



# we put our call backs into a callback list

callbacks = [checkpoint, lr_scheduler]



# Note we use a very small learning rate 

model.compile(loss='categorical_crossentropy',

              optimizer = SGD(lr=.01, momentum=.9),

              metrics = ['accuracy'])



history = model.fit(

    train_dataset,

    epochs = 8,

    callbacks = callbacks,

    validation_data = validation_dataset)



model.save("model_final_Xception.h5")