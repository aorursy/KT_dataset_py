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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")
train = train.iloc[:,0:31]

train.head(10)
test = test.iloc[:,0:30]

test.head(5)
sample.head(5)
from sklearn.preprocessing import LabelEncoder 

# transforma os dados da coluna diagnosis em binario

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])

train.head()
atrib = train.drop('diagnosis', axis=1)

classes = train['diagnosis']
atrib.head(5)
print(atrib.shape)

print(classes.shape)
from sklearn.model_selection import train_test_split



atrib_train, atrib_test, classes_train, classes_test = train_test_split(atrib, classes, test_size=0.25)
print(atrib_train.shape, classes_train.shape, atrib_test.shape, classes_test.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(activation="relu", input_dim=30, units=15, kernel_initializer="uniform"))

model.add(Dense(activation="relu", units=15, kernel_initializer="uniform"))

model.add(Dense(activation="relu", units=15, kernel_initializer="uniform"))

model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(atrib_train, classes_train, batch_size = 10, nb_epoch = 100, validation_data=(atrib_test, classes_test))



score = model.evaluate(atrib_test, classes_test, verbose=0)

print('Erro:', score[0])

print('Precisão:', score[1]*100, "%" )