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

data = pd.read_csv('../input/train.csv')
data.head(5)
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile(optimizer='RMSprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
from keras.utils.np_utils import to_categorical

labels = to_categorical(data.label,10)
train_data = np.asarray(data.drop('label',axis=1))
model.fit(train_data, labels, nb_epoch=20,batch_size = 10000)
print(labels[0:9,:])
np.argmax(labels[0:9,:],axis=1)
print(model.evaluate(train_data[0:9,:],labels[0:9]))
print(np.argmax(model.predict(train_data[0:9,:]),axis=1))