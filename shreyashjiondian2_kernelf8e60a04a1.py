import pandas as pd



file=pd.read_csv('../input/train_labels.csv')

d=dict()

for i in range(220025) :

    d[file.iloc[i][0]]=file.iloc[i][1]
import numpy as np
a=dict()

a['one']=1

print(a)
a['two']=2

print(a)
import cv2

from os.path import join

from os import listdir
feature= []

label = []

path= "../input/train/"

for f in listdir(path):

    print(join(path, f))

    img = cv2.imread(join(path, f))

    resized_img = cv2.resize(img, (48,48))

    feature.append(resized_img)

    label.append(d[f.split(".")[-2]])
feature = np.asarray(feature).reshape(220025,48,48,3)
feature = pd.DataFrame()
len(feature[0])
feature = (feature-128)/255
from keras.models import Sequential

from keras.layers import MaxPooling2D, Conv2D, Flatten, Activation, BatchNormalization, Dense

import keras
model = Sequential()



model.add(Conv2D(32, input_shape=feature.shape[1:], kernel_size=(3,3), padding="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Conv2D(128, kernel_size=(3,3), padding="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Dense(64))



model.add(Flatten())



model.add(Dense(1))



model.compile(loss="binary_crossentropy",

             optimizer=keras.optimizers.Adamax(lr=5, decay=0.1),

             metrics=["acc"])
model.fit(feature, label, epochs=5, batch_size=32, validation_split=0.1)
label = np.asarray(label).reshape(220025,1)