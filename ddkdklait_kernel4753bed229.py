import matplotlib.pyplot as plt

from keras import applications

from keras import optimizers

from keras.layers import Dense

from keras.layers import GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib

import tensorflow as tf





img_width, img_height = 224, 224



train_data_dir = '../input/fruits-360_dataset/fruits-360/Training/'

test_data_dir = '../input/fruits-360_dataset/fruits-360/Test/'

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')
inception_base = applications.ResNet50(weights=None, include_top=False)



x = inception_base.output

x = GlobalAveragePooling2D()(x)



x = Dense(100000, activation='relu')(x)



predictions = Dense(103, activation='softmax')(x)



inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

inception_transfer.compile(loss='categorical_crossentropy',

                           optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

                           metrics=['accuracy'])
with tf.device("/device:GPU:0"):

    history = inception_transfer.fit_generator(

        train_generator,steps_per_epoch = 2000,

        epochs=10, shuffle=True, verbose=1,validation_steps=1000, validation_data=validation_generator)
plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()