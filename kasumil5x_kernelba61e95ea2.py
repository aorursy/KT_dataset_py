import os

import pandas as pd



# Image root directories.

train_dir = '../input/train/train/'

test_dir = '../input/test/test/'



# Read in training CSV.

train_df = pd.read_csv('../input/train.csv')



# Create test DF by hand based on the contents.

test_images = os.listdir(test_dir)

test_df = pd.DataFrame(

    list(zip(test_images, [0] * len(test_images))),

    columns=['id', 'has_cactus']

)



# TEMP

#train_df = train_df[:100]

#test_df = test_df[:100]
from sklearn.model_selection import train_test_split



# Split out the dataframe like usual (but this time with the filepaths and labels rather than raw data).

X = train_df.id

y = train_df.has_cactus

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# Recombine the dataframes for use in the generators.

train_gen_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)

valid_gen_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
from keras.preprocessing.image import ImageDataGenerator



batch_size = 64



train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    horizontal_flip=True,

    vertical_flip=True

)

train_gen = train_datagen.flow_from_dataframe(

    dataframe=train_gen_df,

    directory=train_dir,

    x_col='id',

    y_col='has_cactus',

    class_mode='other',

    batch_size=batch_size,

    target_size=(32, 32)

)



valid_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0

)

valid_gen = valid_datagen.flow_from_dataframe(

    dataframe=valid_gen_df,

    directory=train_dir,

    x_col='id',

    y_col='has_cactus',

    class_mode='other',

    batch_size=batch_size,

    target_size=(32, 32)

)
from keras import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Flatten, Dense



input_shape = (32, 32, 3)



model = Sequential()

model.add(Conv2D(8, (3,3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(16, (3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

#

model.add(Conv2D(32, (3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, (3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

#

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(0.4))

#

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.4))

#

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [

    EarlyStopping(monitor='val_loss', patience=10),

    ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)

]



history = model.fit_generator(

    train_gen,

    validation_data=valid_gen,

    validation_steps=len(train_gen),

    steps_per_epoch=len(train_gen_df) / batch_size, # bug fix

    epochs=100,

    verbose=True,

    shuffle=True,

    callbacks=callbacks

)
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()

plt.show()



plt.plot(history.history['acc'], label='acc')

plt.plot(history.history['val_acc'], label='val_acc')

plt.legend()

plt.show()
model.load_weights('model.h5')



test_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0

)

test_gen = valid_datagen.flow_from_dataframe(

    dataframe=test_df,

    directory=test_dir,

    x_col='id',

    class_mode=None,

    batch_size=1,

    target_size=(32, 32),

    shuffle=False

)
predictions = model.predict_generator(test_gen, steps=len(test_gen), verbose=True)

#predictions_binary = [0 if x < 0.5 else 1 for x in predictions]
test_df['has_cactus'] = predictions

test_df.to_csv('submission.csv', index=False)

print('Done!')
test_df