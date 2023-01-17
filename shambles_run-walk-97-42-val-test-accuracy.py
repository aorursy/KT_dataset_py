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
import pandas as pd
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.models import Sequential, Model
data = pd.read_csv('../input/dataset.csv')
data.info()
x_train = data[['wrist', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']]
y_train = data[['activity']]
x_train.info(), y_train.info()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train.shape, y_train.shape
model = Sequential()
model.add(Dense(250, activation = 'relu', input_shape = (7,)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.25))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 40000, epochs = 100, validation_split = 0.2, verbose = 2)
