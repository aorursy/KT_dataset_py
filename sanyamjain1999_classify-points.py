import numpy as np

import pandas as pd
x = pd.read_csv("../input/Logistic_X_Train.csv")

y = pd.read_csv("../input/Logistic_Y_Train.csv")

xt = pd.read_csv("../input/Logistic_X_Test.csv")

yt = pd.read_csv("../input/SampleOutput.csv")
from matplotlib import pyplot as plt

from keras.models import Sequential,Model,load_model

from keras.layers import Dense,Activation,Input

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print((x.shape, y.shape))

print(xt.shape)
model = Sequential()

model.add(Dense(256, input_dim=(2)))

model.add(Activation('relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(2))

model.add(Activation('softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model.h5", save_best_only=True,monitor='val_acc')

hist = model.fit(x, y, epochs = 100, validation_split = 0.2, shuffle = True, batch_size = 64,callbacks = [checkpoint])
from keras.models import load_model

r_model = load_model("model.h5")
pred = r_model.predict_classes(xt)

pred
from pandas import DataFrame

points = {'label' : pred}

df = DataFrame(points, columns = ['label'])

df.head()
df.to_csv('submission.csv', index=None)