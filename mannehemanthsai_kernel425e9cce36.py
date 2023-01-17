# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
p=os.path.join("../input/skin-cancer-malignant-vs-benign/data/train/")

print(p)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from keras.preprocessing import image

import tensorflow as tf
malignant=os.path.join(p,'malignant')

benign=os.path.join(p,'benign')
BATCH_SIZE=80

IMG_SHAPE=80
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen_train = ImageDataGenerator(rescale=1./255, horizontal_flip=False)

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,

                                                     directory=p,

                                                     target_size=(IMG_SHAPE,IMG_SHAPE),

                                                     class_mode='binary')
p1=os.path.join("../input/skin-cancer-malignant-vs-benign/data/test/")

print(p1)
image_gen_test = ImageDataGenerator(rescale=1./255, horizontal_flip=False)

test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,

                                                     directory=p1,

                                                     target_size=(IMG_SHAPE,IMG_SHAPE),

                                                     class_mode='binary')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()
epochs=40

history = model.fit_generator(

    train_data_gen,

    epochs=epochs,

    validation_data=test_data_gen,

    )
acc,loss=model.evaluate(test_data_gen)
print(acc,loss)
prediction=model.predict(test_data_gen)
print(np.argmax(prediction[0]))
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

    plt.tight_layout()

    plt.show()





augmented_images = [test_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
plt.imshow(train_data_gen[0][0][2])