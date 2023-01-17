import numpy as np

import pandas as pd



import os



import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras import Model

from keras.callbacks import EarlyStopping, TerminateOnNaN, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint



from keras.utils import np_utils



from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)
train_df, val_df = train_test_split(df, train_size=0.7, stratify=df['label'], random_state=42)



train_x = train_df.iloc[:,1:].values.reshape(-1, 28, 28, 1)

train_y = np_utils.to_categorical(train_df.iloc[:, 0])



val_x = val_df.iloc[:,1:].values.reshape(-1, 28, 28, 1)

val_y = np_utils.to_categorical(val_df.iloc[:, 0])



train_x = train_x.astype('float32') / 255.

val_x = val_x.astype('float32') / 255.
datagen = ImageDataGenerator(

    shear_range=0.1,

    zoom_range=0.1,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False,

)

datagen.fit(train_x)
model_input = Input(shape=(28, 28, 1))



layers = Conv2D(64, (5, 5), padding = 'same', activation='relu')(model_input)

layers = MaxPooling2D((2, 2))(layers)



layers = Conv2D(64, (5, 5), padding = 'same', activation='relu')(layers)

layers = MaxPooling2D((2, 2))(layers)



layers = BatchNormalization(momentum=0.5)(layers)

layers = Dropout(0.5)(layers)



layers = Conv2D(64, (3, 3), padding = 'same', activation='relu')(layers)

layers = MaxPooling2D((2, 2))(layers)



layers = Conv2D(64, (3, 3), padding = 'same', activation='relu')(layers)

layers = MaxPooling2D((2, 2))(layers)



layers = Flatten()(layers)



layers = Dense(512, activation='relu')(layers)

layers = BatchNormalization()(layers)

layers = Dropout(0.5)(layers)



layers = Dense(10, activation='softmax')(layers)



model = Model(model_input, layers)



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
callbacks = [

    EarlyStopping(

        monitor='val_loss', 

        mode='min',

        min_delta=0, 

        patience=10, 

        verbose=2,

    ),

    TerminateOnNaN(),

    LearningRateScheduler(lambda epoch_index, learning_rate: learning_rate * 0.9 if 20 > epoch_index > 5 else learning_rate, verbose=1),

    ReduceLROnPlateau(

        monitor='val_loss', 

        mode='min',

        factor=0.5, 

        patience=3, 

        min_lr=0.00005

    ),

]



epochs, batch_size = 50, 256



model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),

    steps_per_epoch=train_x.shape[0] // batch_size,

    epochs=epochs,

    validation_data=(val_x, val_y),

    verbose=2, 

    callbacks=callbacks,

)
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0)



test_x = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.

test_x.shape
predictions = model.predict(test_x).argmax(axis=-1)



result = pd.DataFrame({

    'ImageId': range(1, test_x.shape[0] + 1),

    'Label': predictions,

})

result.to_csv("submission.csv", index=False)