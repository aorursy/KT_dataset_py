# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense,Dropout , Lambda, Flatten

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", comment='#')



train.shape
train.head()
test= pd.read_csv("../input/test.csv", comment='#')
test.shape
test.head()
X_train = (train.iloc[:,1:].values)

y_train = train.iloc[:,0].values

X_test = test.values
X_train = X_train.reshape((42000, 28 * 28))
X_test = X_test.reshape((28000, 28 * 28))
X_train = (X_train -128)/ 128

X_test = (X_test -128)/ 128
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

 loss='categorical_crossentropy',

 metrics=['accuracy'])
history=model.fit(X_train, y_train, validation_split = 0.05, 

            nb_epoch=15, batch_size=128)
predictions = model.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})
submissions.to_csv("sample_submission.csv", index=False, header=True)