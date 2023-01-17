import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train = train.iloc[:,0:31]

test = test.iloc[:,0:30]
from sklearn.preprocessing import LabelEncoder 

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])

train.head()
X = train.drop('diagnosis', axis=1)

y = train['diagnosis']
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
print('shape x_train', x_train.shape)

print('shape x_test', x_test.shape)

print('shape y_train', y_train.shape)

print('shape y_test', y_test.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop



y_train = keras.utils.to_categorical(y_train, 2)

y_test = keras.utils.to_categorical(y_test, 2)

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(30,)))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train,

                    verbose=1,

                    epochs=20,

                    batch_size=351,

                    validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print('Error:', score[0])

print('Precision:', score[1]*100, "%" )