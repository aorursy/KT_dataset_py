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
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
# Check train data
!head /kaggle/input/digit-recognizer/train.csv
# Check test data
!head /kaggle/input/digit-recognizer/test.csv
# Check submission sample
!head /kaggle/input/digit-recognizer/sample_submission.csv
train_dataset = np.loadtxt('/kaggle/input/digit-recognizer/train.csv', skiprows=1, delimiter=',')
train_dataset[0:5]
x_train = train_dataset[:, 1:]
x_train /= 255.0
x_train[:5, 440:450]
y_train = train_dataset[:, 0]
y_train = utils.to_categorical(y_train)
y_train[:5]
model = Sequential()
model.add(Dense(1000, input_dim=784, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2)

history = model.fit(x_train, 
                    y_train, 
                    batch_size=300, 
                    epochs=40,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stopping_callback])

print("\nStop on epoch: ", early_stopping_callback.stopped_epoch)
plt.plot(history.history['acc'], 
         label='Test')
plt.plot(history.history['val_acc'], 
         label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accurancy')
plt.legend()
plt.show()