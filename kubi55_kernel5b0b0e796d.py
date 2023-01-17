# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import SGD



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nb_classes = 10
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')
train.columns
train.head()
X_train=train.drop('label',axis=1).astype("float32")

Y_train=train['label'].astype('int32')

X_train.shape

X_test=test.astype("float32")

                   
from keras.utils import np_utils



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(Y_train, 10)

Y_train.shape


from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train)
from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), 

              metrics=['accuracy'])


network_history = model.fit(X_train, Y_train, batch_size=128, 

                            epochs=10, validation_data=(X_val, Y_val))

def plot_history(network_history):

    plt.figure()

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.plot(network_history.history['loss'])

    plt.plot(network_history.history['val_loss'])

    plt.legend(['Training', 'Validation'])



    plt.figure()

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.plot(network_history.history['accuracy'])

    plt.plot(network_history.history['val_accuracy'])

    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.show()



plot_history(network_history)




predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)