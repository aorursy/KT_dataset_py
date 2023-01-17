import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D



import os

import cv2
DIR = "../input/friendshipgoals/data/test"

test = pd.read_csv("../input/friendshipgoals/data/Test.csv")

test['Category'] = -1

test['Filename'] = test['Filename'].apply(lambda x: os.path.join(DIR, x))

test.head()
train_data = ImageDataGenerator()

train = train_data.flow_from_directory(

    directory="../input/friendshipgoals/data/train/",

    target_size=(128, 128),

)
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))

model.add(MaxPooling2D())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(

    train,

    epochs=35

)
model.save("friendship_goal_model.h5")
idx2cat = {0:'Adults', 1:'Teenagers', 2:'Toddler'}

cat = []

for img_name in test['Filename']:

    img = cv2.imread(img_name)

    img = cv2.resize(img, (128, 128))

    cls = model.predict_classes(np.array([img])).tolist()[0]

    cat.append(idx2cat[cls]) 
new_test = pd.read_csv("../input/friendshipgoals/data/Test.csv")

submission = pd.DataFrame({'Filename':new_test['Filename'], 'Category':cat})

submission.to_csv("friendship_goals_sub.csv", index=False)