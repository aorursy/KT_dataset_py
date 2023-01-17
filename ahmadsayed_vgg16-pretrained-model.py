import keras

from keras.applications import VGG16
vgg = VGG16(include_top = False, weights = "imagenet",

  input_shape = (224, 224, 3))
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.models import Sequential

from keras import backend as K
model  = Sequential()

model.add(vgg)
model.add(Conv2D(512, 3, activation='relu', padding="same"))

model.add(Conv2D(512, 3, activation='relu', padding="same"))

model.add(Conv2D(512, 3, activation='relu', padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(512, 3, activation='relu', padding="same"))

model.add(Conv2D(512, 3, activation='relu', padding="same"))

model.add(Conv2D(512, 3, activation='relu', padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(256, 3, activation='relu', padding="same"))

model.add(Conv2D(256, 3, activation='relu', padding="same"))

model.add(Conv2D(256, 3, activation='relu', padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, 3, activation='relu', padding="same"))

model.add(Conv2D(128, 3, activation='relu', padding="same"))

model.add(Conv2D(128, 3, activation='relu', padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding="same"))

model.add(Conv2D(64, 3, activation='relu', padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, 3, activation='relu', padding="same"))

model.summary()
from tqdm import tqdm

import cv2

import os
names = []

data = []

label = []

for i in tqdm(os.listdir("../input/first_part")):

    sub_folder_path = os.path.join("../input/first_part", i)

    for j in tqdm(os.listdir(sub_folder_path)):

        sub_sub_folder = os.path.join(sub_folder_path, j)

        for ij in os.listdir(sub_sub_folder):

            if ".jpg" in ij:

                img_name = ij.split('.jpg')[0]

                names.append([img_name, sub_sub_folder])
X = []

y = []

for img_info in tqdm(names):

    name = img_info[0]

    orignal_path = img_info[1] + "/" + name + ".jpg"

    seg_path = img_info[1] + "/" + name + "_seg.png"

    img = cv2.imread(orignal_path)

    img = cv2.resize(img, (224, 224))

    seg = cv2.imread(seg_path)

    seg = cv2.resize(seg, (224, 224))

    X.append(img)

    y.append(seg)
import numpy as np
X = np.array(X).reshape(-1, 224, 224, 3)

y = np.array(y).reshape(-1, 224, 224, 3)

print(X.shape)

print(y.shape)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
from keras import optimizers 
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X, y, epochs=100, validation_split=0.2)
model.save("model.h5")