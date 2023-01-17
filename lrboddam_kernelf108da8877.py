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
import numpy as np

from PIL import Image

trnX = np.zeros((60000, 28, 28))

for i in range(trnX.shape[0]):

    img = Image.open("../input/mnist_trn_images/mnist_trn_" + str(i).zfill(5) + ".png")

    trnX[i] = np.asarray(img)



trnY = np.zeros((60000, 1))

input = open("../input/mnist_trn.csv", "r")

header = input.readline()

for i in range(trnY.shape[0]):

     trnY[i,0] = int(input.readline().strip("\r\n").split(",")[1])

input.close()



tstX = np.zeros((10000, 28, 28))

for i in range(tstX.shape[0]):

    img = Image.open("../input/mnist_tst_images/mnist_tst_" + str(i).zfill(5) + ".png")

    tstX[i] = np.asarray(img)



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping



trnX = trnX.reshape(trnX.shape[0], trnX.shape[1] * trnX.shape[2])

tstX = tstX.reshape(tstX.shape[0], tstX.shape[1] * tstX.shape[2])



trnX = trnX.astype("float32")

trnY = trnY.astype("int32")

tstX = tstX.astype("float32")
# the next two lines define a multi-nomial (10 class) logistic regression model

model = Sequential()

model.add(Dense(10, input_shape = (trnX.shape[1],), activation = "softmax"))



# the next four lines define a multi-layer perceptron model (replaces the logistic regression model when uncommented)

model = Sequential()

model.add(Dense(512, input_shape = (trnX.shape[1],), activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation = "softmax"))



model.summary()

model.compile(loss = "sparse_categorical_crossentropy",

              optimizer = RMSprop(),

              metrics = [ "accuracy" ])



model.fit(trnX, trnY, validation_split = 0.1, epochs = 100, callbacks = 

          [ EarlyStopping(monitor = "val_acc", patience = 5, restore_best_weights = True) ])

probabilities = model.predict(tstX)

classes = probabilities.argmax(axis = -1)



predictions = open("../predictions.csv", "w")

predictions.write("id,label\n")

for i in range(tstX.shape[0]):

    predictions.write(str(i).zfill(5) + "," + str(classes[i]) + "\n")

predictions.close()