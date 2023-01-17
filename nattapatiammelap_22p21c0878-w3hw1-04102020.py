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
import cv2

from scipy import signal

import matplotlib.pyplot as plt 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense



def genimg(img):

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)

    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7)

    edge = sobelx+sobely

    edge = np.array([[[j] for j in i] for i in edge])

    return edge



model = Sequential()

model.add(Conv2D(64,(3,3), padding="same", activation="relu", input_shape=(224,224,1)))

model.add(Conv2D(64,(3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(4096,activation="relu"))

model.add(Dense(4096,activation="relu"))

model.add(Dense(2, activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.sparse_categorical_crossentropy)



df = pd.read_csv("/kaggle/input/super-ai-image-classification/train/train/train.csv")

df["category"] = [str(i) for i in df["category"]]



datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=genimg)

batch = 32

train_generator= datagen.flow_from_dataframe(dataframe=df, 

                                             directory="/kaggle/input/super-ai-image-classification/train/train/images", 

                                             x_col="id", 

                                             y_col="category", 

                                             color_mode="grayscale",

                                             class_mode="binary", 

                                             target_size=(224,224),

                                             batch_size=batch)





num_batch = train_generator.n//train_generator.batch_size

with tf.device('/GPU:0'):

    Progbar = keras.utils.Progbar(num_batch)

    for epoch in range(10):

        batches = 0

        for x, y in train_generator:

            history = model.fit(x, y, verbose=1)

            batches += 1

            Progbar.update(batches, values=[('loss', history.history['loss'][0])])

            if batches >= num_batch:

                print(epoch)

                break



Xtest = []

testid = []

for dirname, _, filenames in os.walk('/kaggle/input/super-ai-image-classification/val/val/images'):

    for filename in filenames:

        testid.append(filename)

        img = cv2.imread(os.path.join(dirname, filename))

        img = tf.image.resize(img, [224,224])

        img = np.float32(tf.image.rgb_to_grayscale(img)) 

        Xtest.append(genimg(img))



predictions = model.predict(np.array(Xtest))

prediction = [int(i[1]>i[0]) for i in predictions]

output = pd.DataFrame({'id': testid, 'category': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
