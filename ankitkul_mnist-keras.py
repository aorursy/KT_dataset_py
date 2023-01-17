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
from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout
import pandas as pd

import numpy as np
# Read data

train = pd.read_csv('../input/train.csv')

labels = train.ix[:,0].values.astype('int32')

X_train = (train.ix[:,1:].values).astype('float32')

X_test = (pd.read_csv('../input/test.csv').values).astype('float32')
# convert list of labels to binary class matrix

y_train = np_utils.to_categorical(labels) 
# pre-processing: divide by max and substract mean

scale = np.max(X_train)

X_train /= scale

X_test /= scale



mean = np.std(X_train)

X_train -= mean

X_test -= mean
input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]
input_dim
nb_classes
# Here's a Deep Dump MLP (DDMLP)

model = Sequential()

model.add(Dense(128, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))
# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

print("Training...")

model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)

print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)
def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
write_preds(preds, "keras-mlp.csv")