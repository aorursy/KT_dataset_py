from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 

import pandas as pd

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K

from keras.callbacks import ModelCheckpoint
train_dir = '/kaggle/input/paintings-0/mega_set/train'

val_dir = '/kaggle/input/paintings-0/mega_set/validation'

img_width, img_height = 299, 299

input_shape = (img_width, img_height, 3)

epochs = 10

batch_size = 75
#base_model = InceptionV3(weights='imagenet', include_top=False)

model = InceptionV3(weights=None, include_top=True,classes=7)

# add a global spatial average pooling layer

'''x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 7 classes

predictions = Dense(7, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)'''
'''for layer in model.layers[:205]:

    layer.trainable = False'''
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['cosine_proximity','accuracy'])
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical')
val_generator = datagen.flow_from_directory(

    val_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',)
checkpoint = ModelCheckpoint('/kaggle/working/weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]
callback = model.fit(

    train_generator,

    epochs=epochs,

    validation_data=val_generator,

    callbacks=callbacks_list)
callback.model.save_weights('model.h5')
pred = model.predict_generator(val_generator)

pred_ = np.zeros((pred.shape[0]),dtype=int)

for k in range(len(pred_)):

    pred_[k] = int([i for i, j in enumerate(pred[k]) if j == max(pred[k])][0])

np.bincount(pred_)