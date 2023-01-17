import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from random import seed, shuffle
# Loading data into a convinient format

class_names = []

xtrain = []

ytrain = []

for i,dir_name in enumerate(os.listdir("../input/fruits/fruits-360/Training")):

    class_names.append(dir_name)

    for img_name in os.listdir("../input/fruits/fruits-360/Training/" + dir_name):

        img = mpimg.imread("../input/fruits/fruits-360/Training/" + dir_name + "/" + img_name)

        xtrain.append(img)

        label = np.zeros(131)

        label[i] = 1

        ytrain.append(label)

    



xtrain = np.array(xtrain)

ytrain = np.array(ytrain)



seed(42)

shuffle(xtrain)

seed(42)

shuffle(ytrain)



print("X Train shape: ",xtrain.shape)

print("Y Train shape: ",ytrain.shape)





xtest = []

ytest = []



for i, dir_name in enumerate(os.listdir("../input/fruits/fruits-360/Test")):

    for img_name in os.listdir("../input/fruits/fruits-360/Test/" + dir_name):

        img = mpimg.imread("../input/fruits/fruits-360/Test/" + dir_name + "/" + img_name)

        xtest.append(img)

        label = np.zeros(131)

        label[i] = 1

        ytest.append(label)

        

xtest = np.array(xtest)

ytest = np.array(ytest)



seed(84)

shuffle(xtest)

seed(84)

shuffle(ytest)



print("X test shape: ",xtest.shape)

print("Y test shape: ",ytest.shape)
from tensorflow.keras.layers import Conv2D, Input, Dense, AveragePooling2D, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Sequential

import gc



model = Sequential()

model.add(Input((100,100,3)))

model.add(Flatten())

model.add(Dense(131,activation="softmax"))





opt = Adam()



model.compile(opt,loss="categorical_crossentropy",metrics=["accuracy"])



model.summary()



model.fit(xtrain,ytrain,epochs=10,batch_size=32)



gc.collect()
model.evaluate(xtest[:-3],ytest[:-3])
for img in xtest[-3:]:

    result = model.predict(img.reshape(1,100,100,3))

    plt.title(class_names[result.tolist()[0].index(1)])

    plt.imshow(img)

    plt.show()