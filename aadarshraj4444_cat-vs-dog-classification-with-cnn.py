# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_cat_dir = os.path.join("/kaggle/input/cat-and-dog/training_set/training_set/cats/")

train_dog_dir = os.path.join("/kaggle/input/cat-and-dog/training_set/training_set/dogs/")



validation_cat_dir = os.path.join("/kaggle/input/cat-and-dog/test_set/test_set/cats/")

validation_dog_dir = os.path.join("/kaggle/input/cat-and-dog/test_set/test_set/dogs/")

    
train_cat_names = os.listdir(train_cat_dir)

print(train_cat_names[:5])

train_dog_names = os.listdir(train_dog_dir)

print(train_dog_names[:5])



validation_cat_name = os.listdir(validation_cat_dir)

print(validation_cat_name[:5])

validation_dog_names = os.listdir(validation_dog_dir)

print(validation_dog_names[:5])
print("total training cat images : ", len(os.listdir(train_cat_dir)))

print("total training dog images : ", len(os.listdir(train_dog_dir)))



print("total validation cat images : ", len(os.listdir(validation_cat_dir)))

print("total validation dog images : ", len(os.listdir(validation_dog_dir)))
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Parameter for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



# Index for iterating over images

pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)





pic_index += 8

next_cat_pix = [os.path.join(train_cat_dir, fname) 

                for fname in train_cat_names[pic_index-8:pic_index]]

next_dog_pix = [os.path.join(train_dog_dir, fname) 

                for fname in train_dog_names[pic_index-8:pic_index]]



for i, img_path in enumerate(next_cat_pix+next_dog_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
import tensorflow as tf

from tensorflow import keras
model = keras.Sequential([

    

    keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(300,300,3)),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(32,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(32,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Flatten(),

    keras.layers.Dense(512,activation="relu"),

    keras.layers.Dense(1,activation="sigmoid")

])
model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),

             loss="binary_crossentropy",

             metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)



# flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory("/kaggle/input/cat-and-dog/training_set/training_set/",

                                                   target_size=(300,300),

                                                   batch_size=32,

                                                   class_mode="binary")





# flow test images in batches of 128 using test_datagen generator

test_generator = test_datagen.flow_from_directory("/kaggle/input/cat-and-dog/test_set/test_set/",

                                                 target_size=(300,300),

                                                 batch_size=32,

                                                 class_mode="binary")
history = model.fit(train_generator,

                   steps_per_epoch=8,

                   epochs=50,

                   verbose=1,

                   validation_data=test_generator,

                   validation_steps=8)
import numpy as np

import cv2

from keras.preprocessing import image
path = "/kaggle/input/images/d2.jpeg"





img = image.load_img(path,target_size=(300,300))

x = image.img_to_array(img)

x = np.expand_dims(x,axis=0)



images = np.vstack([x])

classes = model.predict(images,batch_size=10)

print(classes[0])

if classes[0] > 0.5:

    print("This IS A DOG.")

else:

    print("This IS A CAT.")

    

# check the image is cat or dog with open-cv or model predicted right or not

path = "/kaggle/input/images/d2.jpeg"

image.load_img(path)