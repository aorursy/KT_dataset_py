# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import random
import os
import shutil

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
# constant values
VALID_SPLIT = 0.2
IMG_SIZE = 64
BATCH_SIZE = 128
CHANNEL = 1

# iterate over train file

label = []
data = []
counter = 0

data_path = "../input/train/train"

for img in os.listdir(data_path):
    img_data = cv2.imread(os.path.join(data_path, img), cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    if img.startswith("cat"):
        label.append(0)
    elif img.startswith("dog"):
        label.append(1)
    try:
        data.append(img_data/255)
    except:
        label = label[:len(label)-1]

    counter += 1
    if counter % 1000 == 0:
        print(f"Image data retrieved: {counter}")

data = np.array(data)
data = data.reshape((data.shape)[0], (data.shape)[1], (data.shape)[2], 1)
label = np.array(label)

print(f"data shape: {data.shape}")
print(f"label shape: {label.shape}")

from keras.layers import Dropout
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, CHANNEL),
                 activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

from keras.callbacks import ModelCheckpoint

callback_save = ModelCheckpoint("model.h5",
                                monitor='val_loss',
                                verbose=0,
                                save_weights_only=True,
                                mode='auto',
                                save_best_only=True)

# alternate
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(data)

train_history = model.fit_generator(
    datagen.flow(data, label, batch_size=BATCH_SIZE),
    steps_per_epoch=len(data) / BATCH_SIZE,
    epochs=10,
    callbacks=[callback_save])

# test data
test_data=[]
id=[]
counter=0
for file in os.listdir("../input/test1/test1"):
    img_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)
    try:
        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
        test_data.append(img_data/255)
        id.append((file.split("."))[0])
    except:
        print (f"Doesn't work")
    counter+=1
    if counter%3000==0:
        print (f"Image Retreived: {counter}")

test_data=np.array(test_data)
print (test_data.shape)
test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],1)
df_submission=pd.DataFrame({"id":id})
preds=model.predict(test_data)
preds=np.round(preds,decimals=2)
labels=[1 if value>0.5 else 0 for value in preds]

df_submission["label"]=labels
df_submission.info()
df_submission.to_csv("cnn_submission.csv",index=False)


