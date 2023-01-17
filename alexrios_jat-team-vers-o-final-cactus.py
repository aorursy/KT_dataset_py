import tensorflow as tf

from tensorflow import keras

from keras import regularizers, optimizers

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd

import numpy as np
print(tf.__version__)

print(tf.test.is_gpu_available())
train_df=pd.read_csv('../input/aerial-cactus-identification/train.csv', dtype=str)

test_df=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv', dtype=str)
batch_size = 32



datagen_train = ImageDataGenerator(validation_split=0.2, rescale=1./255,

                                   horizontal_flip=True, vertical_flip=True,

                                   rotation_range=0.15, zoom_range=0.15,

                                   shear_range=0.15, width_shift_range=0.2,

                                   height_shift_range=0.2)



train_generator = datagen_train.flow_from_dataframe(

    dataframe=train_df, directory='../input/aerial-cactus-identification/train/train',

    x_col="id", y_col="has_cactus",

    subset="training", batch_size=batch_size,

    target_size=(32,32), class_mode='binary')



datagen_valid=ImageDataGenerator(validation_split=0.2,

                                 rescale=1./255)



valid_generator=datagen_valid.flow_from_dataframe(

    dataframe=train_df,

    directory='../input/aerial-cactus-identification/train/train',

    x_col="id",

    y_col="has_cactus",

    subset="validation",

    batch_size=batch_size,

    target_size=(32,32),

    class_mode='binary')



test_datagen = ImageDataGenerator(rescale=1./255)



test_generator=test_datagen.flow_from_dataframe(

    dataframe=test_df,

    directory='../input/aerial-cactus-identification/test/test',

    x_col="id",

    y_col=None,

    batch_size=1,

    seed=1234,

    shuffle=False,

    class_mode=None,

    target_size=(32,32))
input_shape = (32,32,3)

dropout_dense_layer = 0.6

  

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

              loss=keras.losses.binary_crossentropy,

              metrics=["accuracy"])
checkpoint = ModelCheckpoint(filepath='/kaggle/working/best_model.h5',

                             monitor='val_accuracy', verbose=1, save_best_only=True)

earlystopping = EarlyStopping(monitor='val_accuracy', mode='max',

                              patience=30, restore_best_weights=True, verbose=1)

callbacks_list = [earlystopping, checkpoint]
epochs = 200

history = model.fit_generator(generator=train_generator,

                    validation_data=valid_generator,

                    epochs=epochs, shuffle=True,

                    callbacks=callbacks_list, verbose=1)
model.load_weights('/kaggle/working/best_model.h5')

pred=model.predict_generator(test_generator,verbose=1)

pred_binary = [0 if value<0.50 else 1 for value in pred]  
csv_file = open("submission.csv","w")

csv_file.write("id,has_cactus\n")

for filename, prediction in zip(test_generator.filenames,pred_binary):

    name = filename

    csv_file.write(str(name)+","+str(prediction)+"\n")

csv_file.close()