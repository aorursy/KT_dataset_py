# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

#     # The fifth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron.

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])



# model.compile(loss='binary_crossentropy',

#               optimizer='adam',

#               metrics=['binary_accuracy'])
# importing the images 

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)



test_gen = ImageDataGenerator(

        rescale=1./255

        )



val_gen = ImageDataGenerator(

        rescale=1./255

        )



# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        '../input/chest_xray/chest_xray/train',  # This is the source directory for training images

        target_size=(300, 300),  # All images will be resized to 

    color_mode='grayscale',    

    batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')





testgen = test_gen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(300, 300), color_mode='grayscale',  class_mode='binary', 

         batch_size=624, shuffle=False)
valgen = val_gen.flow_from_directory(directory="../input/chest_xray/chest_xray/val", 

                                      target_size=(300, 300), color_mode='grayscale',  class_mode='binary', 

         batch_size=16, shuffle=False)
train_generator.class_indices
history = model.fit_generator(

     train_generator,

     steps_per_epoch = 163,

     epochs = 10,

     validation_data = valgen,

     validation_steps = 624)
history.history

test_accu = model.evaluate_generator(testgen,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')
from keras.preprocessing import image

# predicting images

pathtest = '../input/chest_xray/chest_xray/test/PNEUMONIA/person117_bacteria_556.jpeg'

imgtest = image.load_img(pathtest, target_size=(300, 300),color_mode='grayscale' )

xtest = image.img_to_array(imgtest)

xtest = np.expand_dims(xtest, axis=0)



images = np.vstack([xtest])

classes = model.predict(images)

print(classes)

if classes[0]>0.5:

    print("pneumonia")

else: 

    print("normal")

 