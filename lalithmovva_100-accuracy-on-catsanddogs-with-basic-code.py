import tensorflow as tf

import os
print(os.listdir("../input/cat-and-dog/"))
TRAIN_DIR = "../input/cat-and-dog/training_set"

VAL_DIR = "../input/cat-and-dog/test_set"



train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,

                                                          rotation_range=40,

                                                          height_shift_range=0.2,

                                                          width_shift_range=0.2,

                                                          zoom_range=0.2,

                                                          shear_range=0.2,

                                                          horizontal_flip=True,

                                                          fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(TRAIN_DIR,

                                                    batch_size=100,

                                                    class_mode='binary',

                                                    target_size=(150,150))





val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,

                                                          rotation_range=40,

                                                          height_shift_range=0.2,

                                                          width_shift_range=0.2,

                                                          zoom_range=0.2,

                                                          shear_range=0.2,

                                                          horizontal_flip=True,

                                                          fill_mode='nearest')



val_generator = train_datagen.flow_from_directory(VAL_DIR,

                                                    batch_size=100,

                                                    class_mode='binary',

                                                    target_size=(150,150))
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])
model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator,

                             epochs=2,

                             validation_data=val_generator)