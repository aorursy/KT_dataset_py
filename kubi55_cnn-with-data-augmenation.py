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
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

train.columns

y_train=train['label']

x_train=np.array(train.drop(['label'], axis=1))

x_train=x_train/255



from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train, 10)

from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train)
from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.001), 

              metrics=['accuracy'])
network_history = model.fit(X_train, Y_train, batch_size=128, 

                            epochs=100, 

                            validation_data=(X_val, Y_val))
import matplotlib.pyplot as plt

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
predictions = model.predict_classes(test/255, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)