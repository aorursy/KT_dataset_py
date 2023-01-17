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
import pandas as pd

from keras.utils import to_categorical



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
y_train = train.label.values

x_train = train.drop('label', axis=1).values
y_train_hot = to_categorical(y_train)
y_train_hot[0], y_train[0]
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.datasets import mnist
model = Sequential([

    Dense(32, input_shape=(784,)),

    Activation('relu'),

    

    Dense(32),

    Activation('relu'),

    

    Dense(100),

    Activation('relu'),

    

        Dense(200),

    Activation('relu'),

    

        Dense(300),

    Activation('relu'),

    

    Dense(32),

    Activation('relu'),

    

    Dense(10),

    Activation('softmax'),

])
model.compile(optimizer='sgd',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.fit(x_train, y_train_hot, epochs=10, batch_size=500)