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
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:11:09 2018

@author: khoefle
"""


import json
from PIL import Image
import cv2
import numpy as np

with open("../input/train.json") as f:
   data = json.load(f)
   

X = []   
Y = []
for element in data:
    img = cv2.resize(np.array(Image.open("../input/train/train/" + element["filename"])),(64,64))
    X.append(np.array(img/255))
    Y.append(element["class"])

from keras.utils import to_categorical


y_train = np.array(to_categorical(Y))
x_train = np.array(X)


from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions = Dense(29, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy',
              metrics=["accuracy"])


hist = model.fit(x_train,
          y_train,
          batch_size=32,
          validation_split=0.2,
          epochs=10)

from matplotlib import pyplot as plt

plt.figure(1)
plt.title("loss")
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["Training Loss","Validation Loss"])
from matplotlib import pyplot as plt

plt.figure(1)
plt.title("Accuracy")
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.legend(["Training Accuracy","Validation Accuracy"])
with open("../input/sandbox.csv") as f:
    data = f.readlines()
    
content = [x.strip() for x in data]

just_names = [x.split(',')[0] for x in content[1:]]
from tqdm import tqdm
results = []
for c in tqdm(just_names):
    img = Image.open("../input/test/test/" + c)
    
    # falls Sie vorverabrietung haben so führen Sie diese hier durch
    img = np.array(img,dtype=np.float32) / 255
    
    res = model.predict(np.array([img]))

    results.append(np.argmax(res[0]))
# Rausschreiben der Ergebnisse zu einer Submission-Datei

with open("submission.csv","w") as f:
    f.write("ID,Class"+'\n')
    for ID,Class in zip(just_names,results):
        f.write(ID + "," + str(Class) + "\n")