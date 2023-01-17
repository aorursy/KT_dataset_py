# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/face-mask-detection/images'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import xml.etree.ElementTree as et

import re

import pandas as pd



dic = {"image": [],"Dimensions": []}

for i in range(1,116):

	dic[f'Object {i}']=[]

print("Generating data in CSV format....")



for file in os.listdir("/kaggle/input/face-mask-detection/annotations"):

    row = []

    xml = et.parse("/kaggle/input/face-mask-detection/annotations/"+file) 

    root = xml.getroot()

    img = root[1].text

    row.append(img)

    h,w = root[2][0].text,root[2][1].text

    row.append([h,w])



    for i in range(4,len(root)):

        temp = []

        temp.append(root[i][0].text)

        for point in root[i][5]:

            temp.append(point.text)

        row.append(temp)

    for i in range(len(row),119):

        row.append(0)

    for i,each in enumerate(dic):

        dic[each].append(row[i])

df = pd.DataFrame(dic)
df.head()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import numpy as np

import argparse

import pandas as pd

import glob

import os

import cv2

import random as rand



image_directories = sorted(glob.glob(os.path.join("/kaggle/input/face-mask-detection/images","*.png")))

j=0

classes = ["without_mask","mask_weared_incorrect","with_mask"]

labels = []

data = []



print("Extracting each data into respective label folders....")

for idx,image in enumerate(image_directories):

    img  = cv2.imread(image)

    #scale to dimension

    X,Y = df["Dimensions"][idx]

    cv2.resize(img,(int(X),int(Y)))

    #find the face in each object

    for obj in df.columns[3:]:

        info = df[obj][idx]

        if info!=0:

            label = info[0]

            info[0] = info[0].replace(str(label), str(classes.index(label)))

            info=[int(each) for each in info]

            face = img[info[2]:info[4],info[1]:info[3]]

            if((info[3]-info[1])>40 and (info[4]-info[2])>40):

                try:

                    face = cv2.resize(face, (224, 224))

                    face = img_to_array(face)

                    face = preprocess_input(face)

                    data.append(face)

                    labels.append(label)

                    if(label=="mask_weared_incorrect"):

                        data.append(face)

                        labels.append(label)



                except:

                    pass





                

print("Done!")
data = np.array(data, dtype="float32")

labels = np.array(labels)

labels
lb = LabelEncoder()

labels = lb.fit_transform(labels)

labels = to_categorical(labels)
aug = ImageDataGenerator(

    zoom_range=0.1,

    rotation_range=25,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest"

    )
baseModel = MobileNetV2(weights="imagenet", include_top=False,

	input_tensor=Input(shape=(224, 224, 3)))



# construct the head of the model that will be placed on top of the

# the base model

headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(64, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(3, activation="softmax")(headModel)



# place the head FC model on top of the base model (this will become

# the actual model we will train)

model = Model(inputs=baseModel.input, outputs=headModel)



# loop over all layers in the base model and freeze them so they will

# *not* be updated during the first training process

for layer in baseModel.layers:

	layer.trainable = False
INIT_LR = 1e-4

EPOCHS = 50

BS = 1

len(data)
(trainX, testX, trainY, testY) = train_test_split(data, labels,

	test_size=0.3, stratify=labels, random_state=42)
print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])



# train the head of the network

print("[INFO] training head...")

H = model.fit(

	aug.flow(trainX, trainY, batch_size=BS),

	steps_per_epoch=len(trainX) // BS,

	validation_data=(testX, testY),

	validation_steps=len(testX) // BS,

	epochs=EPOCHS)
print("[INFO] evaluating network...")

predIdxs = model.predict(testX, batch_size=32)



# for each image in the testing set we need to find the index of the

# label with corresponding largest predicted probability

predIdxs = np.argmax(predIdxs, axis=1)



# show a nicely formatted classification report

print(classification_report(testY.argmax(axis=1), predIdxs,

	target_names=lb.classes_))



# serialize the model to disk

print("[INFO] saving mask detector model...")



# plot the training loss and accuracy

N = EPOCHS

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")

plt.show()