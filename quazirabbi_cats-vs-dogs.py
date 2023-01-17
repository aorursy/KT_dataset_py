# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import pickle

print(tf.__version__)

print(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)))
base_dir ='../input/cat-and-dog'

train_dir = os.path.join(base_dir, 'training_set/training_set')

test_dir = os.path.join(base_dir, 'test_set/test_set')
target_size = (224, 224)

batch_size = 10





train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,

                                                                rotation_range=40,

                                                                width_shift_range=0.2,

                                                                height_shift_range=0.2,

                                                                shear_range=0.2,

                                                                zoom_range=0.2,

                                                                horizontal_flip=True,

                                                                fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(

        train_dir,  # This is the source directory for training images

        target_size= target_size,  # All images will be resized to 150x150

        batch_size=batch_size,

        class_mode='binary')





test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

        test_dir,  # This is the source directory for training images

        target_size= target_size,  # All images will be resized to 150x150

        batch_size=batch_size,

        class_mode='binary')







# vgg preprocessing



train_datagen_vgg16 = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)



train_generator_vgg16 = train_datagen_vgg16.flow_from_directory(directory=train_dir, target_size=target_size, classes=['cats', 'dogs'], batch_size=batch_size)



test_datagen_vgg16 = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)



test_generator_vgg16 = test_datagen_vgg16.flow_from_directory(directory=test_dir, target_size=target_size, classes=['cats', 'dogs'], batch_size=batch_size, shuffle=False)
# plot diagnostic learning curves

import matplotlib.pyplot as plt



def model_performace(history):# plot loss

    plt.figure(figsize=(10, 6))

    plt.subplot(211)

    plt.title('Cross Entropy Loss')

    plt.plot(history['loss'], color='blue', label='train')

    plt.plot(history['val_loss'], color='orange', label='test')

    

    # plot accuracy

    plt.subplot(212)

    plt.title('Classification Accuracy')

    plt.plot(history['accuracy'], color='blue', label='train')

    plt.plot(history['val_accuracy'], color='orange', label='test')

    

    plt.legend()

    plt.show()

    

def model_lr(lr_min,epochs,history):

    lrs = lr_min* (10 ** (np.arange(epochs) / 20))

    loss = np.array(history["loss"])

    plt.figure(figsize=(10, 6))

    plt.semilogx(lrs,loss)

    plt.xlabel("lr")

    plt.ylabel("loss")

    plt.grid(True)

    

    # clearout the nan in loss array

    loss = loss[np.logical_not(np.isnan(loss))]

    print(f'Minimun loss {np.amin(loss)} at lr :{lrs[np.argmin(loss)]}')

    

    
lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))



check_point_path_accuracy = './custom_model_weights_accuracy.ckpt'

check_point_path_val_accuracy = './custom_model_weights_val_accuracy.ckpt'



model_checkpoint_callback_accuracy = tf.keras.callbacks.ModelCheckpoint(

    check_point_path_accuracy, monitor='vaccuracy', verbose=1, save_best_only=True,

    save_weights_only=True, mode='max', save_freq='epoch'

)



model_checkpoint_callback_val_accuracy = tf.keras.callbacks.ModelCheckpoint(

    check_point_path_val_accuracy, monitor='val_accuracy', verbose=1, save_best_only=True,

    save_weights_only=True, mode='max', save_freq='epoch'

)



def custom_model(lr=1e-8):

        # First Convolution and Max Pooling

        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=train_generator.image_shape),

            tf.keras.layers.MaxPooling2D(2, 2),

            # Second Convolution and Max Pooling

            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same',),

            tf.keras.layers.MaxPooling2D(2,2),

            # Third Convolution and Max Pooling

            tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same',),

            tf.keras.layers.MaxPooling2D(2,2),

            # Flatten the resutls

            tf.keras.layers.Flatten(),

            # Dropout Layer

            tf.keras.layers.Dropout(0.2),

            # Dense Layer

            tf.keras.layers.Dense(512, activation='relu'),

            # Output Layer

            tf.keras.layers.Dense(1,activation='sigmoid')

            ])

        

        opt = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

        return model
def custom_model_2():

        # First Convolution and Max Pooling

        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),

            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),

            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(units=2, activation='softmax')

            ])

        

        opt = tf.keras.optimizers.Adam(lr=1e-4,)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        return model
def custom_model_training_debug(epochs=2):

    model = custom_model()

    history = model.fit_generator(

                                train_generator, 

                                steps_per_epoch=50,

                                validation_data=test_generator, 

                                validation_steps=10,

                                epochs=epochs, 

                                verbose=2,

                                callbacks=[lr_schedule])

    with open('./history_custom_debug.pickle', 'wb') as handle:

        pickle.dump(history.history, handle)

    

    model_lr(1e-8,epochs,history.history)

    return history





# history_custom_debug = custom_model_training_debug(500)
# with open('../input/history-custom-model-training-debug/history_custom_debug.pickle', 'rb') as handle:

#         history_custom_model_training_debug = pickle.load(handle)

        

# model_lr(1e-8,500,history_custom_model_training_debug)

# plt.show()



def custom_model_training(epochs=2):

    model = custom_model(lr = 0.04466835921509635)

    history = model.fit_generator(

                                train_generator, 

                                steps_per_epoch=50,

                                validation_data=test_generator, 

                                validation_steps=10,

                                epochs=epochs, 

                                verbose=2,

                                callbacks=[model_checkpoint_callback])

    with open('./history_custom.pickle', 'wb') as handle:

        pickle.dump(history.history, handle)

    

    return history



# Not enough gpu quota

# custom_model_training(epochs=200)

def custom_model_training_2(epochs=2):

    model = custom_model_2()

    history = model.fit_generator(

                                train_generator_vgg16, 

                                steps_per_epoch=50,

                                validation_data=test_generator_vgg16, 

                                validation_steps=10,

                                epochs=epochs, 

                                verbose=2,

                                callbacks=[model_checkpoint_callback_accuracy,model_checkpoint_callback_val_accuracy])

    with open('./history_custom_model_2.pickle', 'wb') as handle:

        pickle.dump(history.history, handle)

    

    return history



custom_model_training_2(epochs=100)