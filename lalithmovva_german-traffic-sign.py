import os 

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
print(os.listdir("../input/gtsrb-german-traffic-sign/"))
direct = "../input/gtsrb-german-traffic-sign/train"

test_dir = "../input/gtsrb-german-traffic-sign/test"
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./ 255.0,

    validation_split = 0.2,

    rotation_range = 20,

    height_shift_range = 0.15,

    width_shift_range = 0.15,

    zoom_range = 0.2,

    shear_range = 0.2,

    horizontal_flip = True

)
train_generator = train_gen.flow_from_directory(

    direct,

    target_size = (224, 224),

    batch_size = 32,

    class_mode = 'categorical',

    subset = 'training'

)



val_generator = train_gen.flow_from_directory(

    direct,

    target_size = (224, 224),

    batch_size = 32,

    class_mode = 'categorical',

    subset = 'validation'

)
base_model = tf.keras.applications.ResNet50(include_top = False,

                                           input_shape=(224, 224, 3),

                                           weights = 'imagenet')

base_model.trianable = False



model = tf.keras.models.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(43, activation='softmax')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit_generator(train_generator, epochs=20, validation_data = val_generator)
model.save('weights1.h5')
base_model.trainable = True

history1 = model.fit_generator(train_generator, epochs=5, validation_data = val_generator)
model.save('weights2.h5')