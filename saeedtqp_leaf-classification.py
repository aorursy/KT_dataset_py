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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



## Importing sklearn libraries



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



## Keras Libraries for Neural Networks



from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping
## Read data from the CSV file



data = pd.read_csv('../input/leaf-classification/train.csv')

parent_data = data.copy()    ## Always a good idea to keep a copy of original data

ID = data.pop('id')



y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

print(y.shape)



X = StandardScaler().fit(data).transform(data)

print(X.shape)



y_cat = to_categorical(y)

print(y_cat.shape)
# Model



model = Sequential()

model.add(Dense(1500,input_dim=192,  kernel_initializer  = 'uniform', activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1500, activation='sigmoid'))

model.add(Dropout(0.1))

model.add(Dense(99, activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



early_stopping = EarlyStopping(monitor='val_loss', patience=280)

history = model.fit(X,y_cat,batch_size=192,

                    epochs=800 ,verbose=0, validation_split=0.1, callbacks=[early_stopping])

                    
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))
test = pd.read_csv('../input/leaf-classification/test.csv')

index = test.pop('id')

test = StandardScaler().fit(test).transform(test)

yPred = model.predict_proba(test)



yPred = pd.DataFrame(yPred,index=index,columns=sorted(parent_data.species.unique()))



fp = open('submission_nn_kernel.csv','w')

fp.write(yPred.to_csv())