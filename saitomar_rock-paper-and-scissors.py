import os

import tensorflow as tf

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

from multiprocessing import Pool

import matplotlib.pyplot as plt
def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
data_dir = '../input/rockpaperscissors/rps-cv-images/'

augs_gen = ImageDataGenerator(

    rescale=1./255,        

    horizontal_flip=True,

    height_shift_range=.2,

    vertical_flip = True,

    validation_split = 0.2

)  



train_gen = augs_gen.flow_from_directory(

    data_dir,

    target_size = (224,224),

    batch_size=32,

    class_mode = 'categorical',

    shuffle=True,

)



val_gen = augs_gen.flow_from_directory(

    data_dir,

    target_size=(224,224),

    batch_size=32,

    class_mode='categorical',

    shuffle=False,

    subset = 'validation'

)
model = tf.keras.models.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3), use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 2

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 3

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 4

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # layer 5

    tf.keras.layers.Dense(512, activation = 'relu', use_bias=True),

    # layer 6

    tf.keras.layers.Dense(3, activation='softmax', use_bias=True)

    

])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_gen, epochs=20, validation_data = val_gen, verbose = 1)

show_final_history(history)
model_l1 = tf.keras.models.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3), use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 2

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 3

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=0.01) ),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 4

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True,kernel_regularizer =tf.keras.regularizers.l1( l=0.01)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # layer 5

    tf.keras.layers.Dense(512, activation = 'relu', use_bias=True, kernel_regularizer =tf.keras.regularizers.l1( l=0.01)),

    # layer 6

    tf.keras.layers.Dense(3, activation='softmax', use_bias=True)

    

])
model_l1.summary()
model_l1.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history_l1 = model_l1.fit(train_gen, epochs=20, validation_data = val_gen, verbose = 1)

show_final_history(history_l1)
model_l2 = tf.keras.models.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3), use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 2

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 3

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True, kernel_regularizer =tf.keras.regularizers.l2( l=0.01)),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 4

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True , kernel_regularizer =tf.keras.regularizers.l2( l=0.01)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # layer 5

    tf.keras.layers.Dense(512, activation = 'relu', use_bias=True,  kernel_regularizer =tf.keras.regularizers.l2( l=0.01)),

    

    # layer 6

    tf.keras.layers.Dense(3, activation='softmax', use_bias=True)

    

])
model_l2.summary()

model_l2.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history_l2 = model_l2.fit(train_gen, epochs=20, validation_data = val_gen, verbose = 1)

show_final_history(history_l2)

model_drop = tf.keras.models.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3), use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 2

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 3

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True),

    tf.keras.layers.Dropout( 0.2),

    tf.keras.layers.MaxPooling2D(2,2),

    # layer 4

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', use_bias=True),    

    tf.keras.layers.Dropout( 0.2), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # layer 5

    tf.keras.layers.Dense(512, activation = 'relu', use_bias=True),

    tf.keras.layers.Dropout( 0.2),

    # layer 6

    tf.keras.layers.Dense(3, activation='softmax', use_bias=True)

    

])
model_drop.summary()

model_drop.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history_drop = model_drop.fit(train_gen, epochs=20, validation_data = val_gen, verbose = 1)

show_final_history(history_drop)
