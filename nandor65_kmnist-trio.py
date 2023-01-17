# This kernel may help you how to build and and train more than one single model and re-use weights.

# Additionally I try to show why to use more that one CNN to chase higher accuracy points and 

# to reduce stochastic effect of the training process. 



# Version 1 is the original kernel training three CNNs. 

# Run time is approx 2800 sec using GPU.

# Submitted accuracy against test set is 0.9888.



# Version 4 is a 2nd training of the three CNNs and it was not submitted, weights were saved instead. 

# Run time is approx 2800 sec using GPU.



# GPU was switched off at this point.

# Run time for commits below are between 30-60 seconds.



# Version 5 is a no training kernel, it loads weight sets saved in Kernel 4 and creates submission 

# file using all three CNNs.

# Submitted ensemble accuracy against test set is 0.9878.



# Version 6-8 are kernels using single CNN weight sets which were determined in kernel Version 4.

# Single CNN submission accuracies are 0.985 0.9842 and 0.9862.



# Please note variance of single CNN accuracies. 

# Difference between minimum and maximum accuracy is 0.0020 for this model with these hyper parameters. 

# Please also note that ensemble CNN accuracy is significantly higher than any of the accuracies for 

# single CNN ones.



# There is a variance in accuracy even in the case when more than one single CNN is used for prediction. 

# See kernels Version 1 and Version 5 with accucacies of 0.9888 vs 0.9878.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test5000 = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")

print(train.shape)

print(test5000.shape)
from keras.utils.np_utils import to_categorical



X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_train = X_train.values.reshape(-1,28,28,1)



Y_trainlabel = train["label"]

Y_train = to_categorical(Y_trainlabel, num_classes = 10)



X_test5000 = test5000.drop(labels = ["id"],axis = 1)

X_test5000 = X_test5000 / 255.0

X_test5000 = X_test5000.values.reshape(-1,28,28,1)



x_train = X_train

y_train = Y_train



print(x_train.shape)

print(y_train.shape)

print(X_test5000.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization

from keras.initializers import he_normal

from keras.optimizers import Adam



init = he_normal(seed=82)

nets = 3



model = [0] *(nets+1)



for j in range(nets+1):

    model[j] = Sequential()



    model[j].add(Conv2D(32, kernel_size=3, activation='relu' , kernel_initializer=init, input_shape=(28, 28, 1)))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size=3, activation='relu' , kernel_initializer=init ))

    model[j].add(BatchNormalization())



    model[j].add(Conv2D(32, kernel_size=5, activation='relu', kernel_initializer=init ))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=5, activation='relu', kernel_initializer=init ))

    model[j].add(BatchNormalization())



    model[j].add(Conv2D(64, kernel_size=7, activation='relu', kernel_initializer=init ))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size=7, activation='relu', kernel_initializer=init ))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))



    model[j].add(Conv2D(128, kernel_size=4, activation='relu', kernel_initializer=init ))

    model[j].add(BatchNormalization())

    model[j].add(Flatten())

    model[j].add(Dropout(0.4))

    model[j].add(Dense(10, activation='softmax'))



    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



model[0].summary()
for j in range(nets):

    loadmodelname = "../input/kmnist-trio/weights_N" + str(j)

    model[j].load_weights(loadmodelname)

    savemodelname = "weights_N" + str(j)

    model[j].save_weights(savemodelname)

    print("Saving", loadmodelname, "back to", savemodelname)
from keras.preprocessing. image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)

from sklearn.model_selection import train_test_split

#from keras.callbacks import *

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)



nets2train = 0

history = [0] * nets2train

epoks = 35



for j in range(nets2train):

    rs = 10 * j + 1

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, train_size = 0.95, random_state = rs)

    print (X_train2.shape,Y_train2.shape, X_val2.shape, Y_val2.shape)

    modelfilename = "weights_N" + str(j)

    checkpoint = ModelCheckpoint(modelfilename, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, period=epoks)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size = 64), epochs = epoks, steps_per_epoch = X_train2.shape[0]//64, validation_data = (X_val2, Y_val2), callbacks=[annealer, checkpoint], verbose=0)

    print("CNN", j,": Training done.")
results5000 = np.zeros( (X_test5000.shape[0], 10) ) 

nets4predict = 3

allthree = False

for j in range(nets4predict):

    if allthree or j== 2:

        print("CNN",j)

        loadmodelname = "weights_N" + str(j)

        model[j].load_weights(loadmodelname)

        results5000 = results5000 + model[j].predict(X_test5000)

results5000 = np.argmax(results5000,axis = 1)



submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = results5000

submission.to_csv("submission.csv",index=False)



print("DONE.")