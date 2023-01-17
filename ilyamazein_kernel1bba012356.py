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


import random

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, BatchNormalization

from keras.optimizers import RMSprop, SGD

from keras.callbacks import EarlyStopping





trainvaluesfile=open('../input/etrainvalues.txt', 'r')

trainlabelsfile=open('../input/etrainlabels.txt', 'r')



trainvalues=trainvaluesfile.read().split()

for i in range (len(trainvalues)):

    trainvalues[i]=[list(trainvalues[i][j:j+4]) for j in range(101)]

    #trainvalues[i]=list(trainvalues[i])

    

trainlabels=trainlabelsfile.read().split()

for i in range (len(trainlabels)):

    trainlabels[i]=list(trainlabels[i])            



trainvalues=np.array(trainvalues, dtype=float)

#trainvalues=trainvalues.reshape(-1, 101, 4, 1)               

trainlabels=np.array(trainlabels, dtype=float)







model=Sequential()



model.add(Conv1D(filters=40, kernel_size=4, padding='Same', activation ='relu', input_shape=(101, 4)))

model.add(MaxPool1D(pool_size=2))



model.add(Conv1D(filters=60, kernel_size=4, padding='Same', activation ='relu'))

model.add(MaxPool1D(pool_size=2))



model.add(Conv1D(filters=80, kernel_size=3, padding='Same', activation ='relu'))

model.add(MaxPool1D(pool_size=2))



#model.add(BatchNormalization())

#model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(2, activation='sigmoid'))



model.compile(optimizer=SGD(), loss='mean_squared_error', metrics=['accuracy'])



stop2 = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)



history=model.fit(trainvalues, trainlabels, epochs=100, verbose=1,  batch_size=100, validation_split=0.2)





plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()





trainvaluesfile.close()

trainlabelsfile.close()