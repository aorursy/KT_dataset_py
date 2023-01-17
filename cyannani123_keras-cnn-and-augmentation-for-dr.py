import os

import numpy as np

from keras.utils import np_utils

import pandas as pd

from keras import Sequential

from keras.preprocessing import image

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



X_train = train_data.drop(labels = ['label'],axis = 1).values.reshape(len(train_data),28,28,1)

y_train = np_utils.to_categorical(train_data['label'])

X_test = test_data.values.reshape(len(test_data),28,28,1)
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3),input_shape=(28,28,1),activation="relu"))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3),activation="relu"))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3),activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=(3, 3),input_shape=(28,28,1),activation="relu"))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3),activation="relu"))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3),activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(512,activation="relu"))

model.add(BatchNormalization())

model.add(Dense(256,activation="relu"))

model.add(BatchNormalization())

model.add(Dense(128,activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(10,activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #compile model

model.summary()
BATCH_SIZE = 128

img_gen = image.ImageDataGenerator(

    data_format="channels_last",

    rescale=1/255,

    validation_split=0.10,

    rotation_range=10,

    zoom_range = 0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.10

)



img_gen.fit(X_train)



train_generator = img_gen.flow(

    x = X_train, 

    y= y_train,

    subset="training",

    batch_size=BATCH_SIZE,

    shuffle=True

)



validation_generator = img_gen.flow(

    x = X_train, 

    y= y_train,

    subset="validation",

    batch_size=BATCH_SIZE,

    shuffle=True

)



img_gen2 = image.ImageDataGenerator(

    data_format="channels_last",

    rescale=1/255

)



test_generator = img_gen2.flow(

    x = X_test,

    y= None,

    batch_size=1,

    shuffle=False

)
filepath = 'ModelCheckpoint.h5'



callbacks = [

    ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),

    EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    ]



history = model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    epochs=50,validation_data=validation_generator,

    validation_steps=len(validation_generator),

    verbose=2,

    callbacks=callbacks

)
model.load_weights('ModelCheckpoint.h5')



y_pred = model.predict_generator(

    test_generator,

    steps=len(test_data)

)



submission = pd.DataFrame()

submission['ImageId'] = test_data.index.values + 1

submission['Label'] = y_pred.argmax(axis=-1)

submission.to_csv('submission.csv', header=True, index=False)