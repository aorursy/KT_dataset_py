# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from tensorflow.keras.optimizers import RMSprop

import os

from os import listdir, makedirs

from os.path import join, exists, expanduser



from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras import backend as K

import tensorflow as tf

# Any results you write to the current directory are saved as output.
img_width, img_height = 224, 224 # we set the img_width and img_height according to the pretrained models we are

# going to use. The input size for ResNet-50 is 224 by 224 by 3.



train_data_dir = '../input/fruits/fruits-360/Training'

validation_data_dir = '../input/fruits/fruits-360/Test'

nb_train_samples = 31688

nb_validation_samples = 10657

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    rotation_range=5,

    width_shift_range=0.2)



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')
all_labels=train_generator.class_indices

print((all_labels))
model = tf.keras.models.Sequential([  ## initializing and making an empty model with sequential

  

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution layer

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224,3)), ## image input shape is 300x300x3 

                           #16 neurons in this layer





    tf.keras.layers.MaxPooling2D(2,2),    # doing max_pooling

    tf.keras.layers.Dropout(0.2),



  

    # The second convolution layer

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # another layer with 32 neurons

    tf.keras.layers.MaxPooling2D(2,2),     # doing max_pooling

    tf.keras.layers.Dropout(0.2),





    # The third convolution layer

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling

    tf.keras.layers.Dropout(0.2),







    # The fourth convolution layer

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),          # doing max_pooling

    tf.keras.layers.Dropout(0.2),  





    # The fifth convolution 

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling

    tf.keras.layers.Dropout(0.2),







    tf.keras.layers.Flatten(),  # reducing layers arrays 

    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer







    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 

    # 1 for ('pneumonia') class

    tf.keras.layers.Dense(131, activation='softmax')



])



# to get the summary of the model

model.summary()  # summarising a model



# configure the model for traning by adding metrics

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy']) # compiling model
# checkpoint

from keras.callbacks import ModelCheckpoint

filepath="weights.hdf5" # mentioning a file for saving checkpoint model in case it gets interrupted



checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

## we marked filepath, monitor and mentioned to save best model only 





callbacks_list = [checkpoint]  # customising model to save checkpoints

# model.load_weights('../input/fruit-360-90/weights.best.hdf5')




hist = model.fit_generator(

    generator = train_generator,

    steps_per_epoch = 67692//(500),

    epochs = 100,

    validation_data = validation_generator,

    callbacks=callbacks_list,

    validation_steps = 67692 // 500,

    shuffle=True

                   )



model.save_weights('fruits_shuffled_weights.h5')
eval_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = eval_datagen.flow_from_directory(

    '../input/fruits/fruits-360/Test',

    target_size = (224, 224),

    class_mode = 'categorical'

    

)
eval_result = model.evaluate_generator(test_generator)

print('loss rate at evaluation data :', eval_result[0])

print('accuracy rate at evaluation data :', eval_result[1])
Y_pred = model.predict(test_generator)
def corrector(img):

    img = cv2.imread(img)

    img = cv2.resize(img,(224,224))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = np.expand_dims(img,axis=0)

    return img

    
print(all_labels)
p=model.predict(corrector('../input/fruits/fruits-360/Training/Corn Husk/102_100.jpg'))
print(p.argmax(axis=1))
print(list(all_labels.keys())[list(all_labels.values()).index(24)]) 
print(p.argmax(axis=-1))