# 

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import numpy

# fix random seed for reproducibility

seed = 23

numpy.random.seed(seed)

# load pima indians dataset

dataset = numpy.loadtxt("../input/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables

X = dataset[:,0:8]

Y = dataset[:,8]

# create model

model = Sequential()

model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

# Fit the model

model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
# Note here, the file appears to be in ../working

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))
# This part loads the weights saved above

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import numpy

# fix random seed for reproducibility

seed = 23

numpy.random.seed(seed)

# create model

model = Sequential()

model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# load weights

model.load_weights("../working/weights.best.hdf5")

# Compile model (required to make predictions)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Created model and loaded weights from file")

# load pima indians dataset

dataset = numpy.loadtxt("../input/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables

X = dataset[:,0:8]

Y = dataset[:,8]

# estimate accuracy on whole dataset using loaded weights

scores = model.evaluate(X, Y, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))