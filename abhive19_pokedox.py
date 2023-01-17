import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
from keras.layers import Dense , Dropout, Flatten ,GlobalAveragePooling2D

from keras.models import Model

from keras.applications.resnet50 import ResNet50

import matplotlib.pyplot as plt 

import cv2

from PIL import Image

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy

from keras.preprocessing import image
generator = image.ImageDataGenerator(horizontal_flip=True, rotation_range=10, shear_range=.1)
gen = generator.flow_from_directory("../input/dataset/dataset/", target_size=(200, 200), batch_size=32)
img,val=gen.next()
print(val[0])

plt.imshow(img[0])
resnet_mod.summary()
#flat_input = Flatten()(resnet_mod.output)
resnet_mod.output.shape
flat_input = GlobalAveragePooling2D()(resnet_mod.output)

layer1 = Dense(1024,activation = "tanh")(flat_input)

layer2 = Dense(512,activation = "tanh")(layer1)

layer3 = Dense(256,activation = "tanh")(layer2)

layer4 = Dense(128,activation = "tanh")(layer3)

output = Dense(149,activation = "softmax")(layer4)
model = Model(inputs=[resnet_mod.input], outputs=[output])
for layer in resnet_mod.layers:

  layer.trainable = False
resnet_mod.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit_generator(gen, steps_per_epoch=100, epochs=10)
model.evaluate_generator(gen, steps= 10)