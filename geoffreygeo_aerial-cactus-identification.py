# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import PIL.Image as Image

import seaborn as sns

sns.set(style="darkgrid")



import os

print(os.listdir("../input"))







from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pylab as plt



import tensorflow as tf

tf.enable_eager_execution()



import tensorflow_hub as hub

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

submission=pd.read_csv('../input/sample_submission.csv')
print(df['has_cactus'].value_counts())

sns.countplot(x='has_cactus',data=df)
print(tf.test.is_gpu_available())
print(len(os.listdir("../input/train/train")))


train_dir="../input/train/train"

test_dir="../input/test/test"



print("The Training Dir is {}\nThe Validataion Dir is {}".format(len(train_dir),len(test_dir)))
image = Image.open(r"../input/train/train/097480900e80806b84d5caa082eb34d1.jpg")

display(image)

image = np.array(image)

display(image.shape)
df.has_cactus=df.has_cactus.astype(str)
BATCH_SIZE=100

IMG_SHAPE  = 224  # Our training data consists of images with width of 150 pixels and height of 150 pixels

train_generator     = ImageDataGenerator(rescale=1./255,

                                               rotation_range=40,

                                                            width_shift_range=0.2,

                                                            height_shift_range=0.2,

                                                            shear_range=0.2,

                                                            zoom_range=0.2,

                                                            horizontal_flip=True,

                                                            fill_mode='nearest')  # Generator for our training data

test_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
train_data=train_generator.flow_from_dataframe(dataframe=df[:15001],directory=train_dir,x_col='id',

                                            y_col="has_cactus",class_mode='binary',batch_size=BATCH_SIZE,

                                            target_size=(IMG_SHAPE,IMG_SHAPE))
train_data[0][1].shape
validation_data=test_generator.flow_from_dataframe(dataframe=df[15000:],directory=train_dir,x_col='id',

                                            y_col="has_cactus",class_mode='binary',batch_size=BATCH_SIZE,

                                            target_size=(224,224))
validation_data[0][1].shape
#loading the state of art neural network 

# URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/3"



# IMAGE_RES = 224



# feature_extractor =  hub.Module(URL)
#Freezing so that the training modeifies only the final layer

#feature_extractor.trainable = False
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])

model.summary()
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
EPOCHS = 10

history = model.fit_generator(

    train_data,steps_per_epoch=10,

    epochs=EPOCHS,

    validation_data=validation_data,

    validation_steps =20)
history.history
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.savefig('./foo.png')

plt.show()
import PIL.Image as Image





test_image = os.listdir(train_dir)



print(test_image[0])
#image =tf.keras.utils.get_file('655c71d8c3f3d61f3797545e7d0414ce.jpg',train_dir+'655c71d8c3f3d61f3797545e7d0414ce.jpg')

image = Image.open(train_dir+'/'+test_image[0]).resize((IMG_SHAPE,IMG_SHAPE),3)

image
if image.mode != "RGB":

    image = image.convert("RGB")



image = image.resize((224,224))

image= np.array(image)

print(image[np.newaxis, ...].shape)
prediction = model.predict(image[np.newaxis, ...])

print(max(prediction))