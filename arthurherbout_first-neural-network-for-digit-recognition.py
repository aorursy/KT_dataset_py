# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', header = 0)

test = pd.read_csv('../input/test.csv', header = 0)
# Image soze 

img_rows, img_cols = 28, 28

# Number of pixels

num_pixels = img_rows * img_cols

# Size of mini batch

batch_size = 128

# Number of epochs

nb_epoch = 20
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout
# creating a vector of labels, then a matrix of labels

Y = train['label']

Y = np_utils.to_categorical(Y)



nb_classes = Y.shape[1]
print(Y)
train = train.drop(['label'], axis=1)

train = train.values

print(train)
test = test.values

print(test)
nb_classes = Y.shape[1]
# Creating a sequetial model

model = Sequential()

# Adding the input layer

model.add(Dense(785, input_dim = num_pixels, init="normal", activation="relu"))

model.add(Dropout(0.2))

# Adding the output layer

model.add(Dense(nb_classes, init="normal", activation="softmax"))
# Compiling the model

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Printing the model summary

print(model.summary())
# Fitting the model

model.fit(train, Y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
# Making predictions

predictions = model.predict(test)
# Converting the categories to the labels

predictions = np_utils.categorical_probas_to_classes(predictions)
print(predictions)
# Writing data to the output

out = np.column_stack((range(1, predictions.shape[0]+1), predictions))

np.savetxt("submission_v1.csv", out, header="ImageId,Label", comments="", fmt="%d,%d")
print(out)
print(check_output(["ls", "../input"]).decode("utf8"))