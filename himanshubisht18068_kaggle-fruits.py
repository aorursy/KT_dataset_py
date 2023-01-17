import os

import cv2

import keras

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
srcTrain = [

        "../input/fruits-360_dataset/fruits-360/Training/Apple Red 1/",

        "../input/fruits-360_dataset/fruits-360/Training/Banana/",

        "../input/fruits-360_dataset/fruits-360/Training/Pineapple/",

        "../input/fruits-360_dataset/fruits-360/Training/Orange/",

        "../input/fruits-360_dataset/fruits-360/Training/Strawberry/"

      ]

nClasses = len(srcTrain)



srcTest = [

        "../input/fruits-360_dataset/fruits-360/Test/Apple Red 1/",

        "../input/fruits-360_dataset/fruits-360/Test/Banana/",

        "../input/fruits-360_dataset/fruits-360/Test/Pineapple/",

        "../input/fruits-360_dataset/fruits-360/Test/Orange/",

        "../input/fruits-360_dataset/fruits-360/Test/Strawberry/"

    ]
index = 0

xTrain = []

yTrain = np.array([])



for fruitClasses in srcTrain:

    fruitsList = os.listdir(fruitClasses)

    yTrain = np.append(yTrain, np.ones(len(fruitsList), dtype=np.int8) * index)

    index += 1

    

    for fruit in fruitsList:

        img = cv2.cvtColor(cv2.imread(fruitClasses + fruit, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        xTrain.append(img)       

        

        

xTrain = np.array(xTrain, dtype = np.int8)

yTrain = keras.utils.to_categorical(yTrain, nClasses)

xTrain, xValidation, yTrain, yValidation = train_test_split(xTrain, yTrain, shuffle=True, test_size=0.2)

print("Train Data Len : {}".format(len(xTrain)))

print("Validation Data Len : {}".format(len(xValidation)))



index = 0

xTest = []

yTest = np.array([])



for fruitClasses in srcTest:

    fruitsList = os.listdir(fruitClasses)

    yTest = np.append(yTest, np.ones(len(fruitsList), dtype=np.int8) * index)

    index += 1

    

    for fruit in fruitsList:

        img = cv2.cvtColor(cv2.imread(fruitClasses + fruit, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        xTest.append(img)       

        

        

xTest = np.array(xTest, dtype = np.int8)

yTest = keras.utils.to_categorical(yTest, nClasses)

xTest, _, yTest, _ = train_test_split(xTest, yTest, shuffle=True, test_size=0.0)

print("Test Data Len : {}".format(len(xTest)))
from keras import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Activation, Flatten
model = Sequential()

model.add(Conv2D(32, (3,3), strides=1, padding='same', input_shape = (100, 100, 3)))

model.add(Activation('tanh'))

model.add(Conv2D(64, (3,3), strides=1, padding='valid'))

model.add(Activation('relu'))

model.add(Conv2D(128, (3,3), strides=1, padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dense(nClasses))

model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001)

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
batchSizeTrain = 50

nEpochs = 5

model.fit(xTrain, yTrain, batch_size=batchSizeTrain, epochs=nEpochs, validation_data=(xValidation, yValidation))
batchSizeTest = 100

scores = model.evaluate(xTest, yTest, verbose=1, batch_size=batchSizeTest)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt



def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None): 

    if type(ims[0]) is np.ndarray:

        ims = np.array(ims).astype(np.uint8)

        if (ims.shape[-1] != 3):

            ims = ims.transpose((0,2,3,1))

    f = plt.figure(figsize=figsize)

    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):

        sp = f.add_subplot(rows, cols, i+1)

        sp.axis('Off')

        if titles is not None:

            sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i], interpolation=None if interp else 'none')

predictions = model.predict(xTest, batch_size=50)

plots(xTest[10:20], titles=np.argwhere(yTest[10:20] == 1)[:,1])
d = np.argwhere(yTest[:10] == 1)[:,1]

print(d)