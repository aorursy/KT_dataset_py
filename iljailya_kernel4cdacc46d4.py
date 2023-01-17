# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow.keras as keras





from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
val = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")
del test['id']
X = train.iloc[:,1:].values

y = train.iloc[:,0].values

x_val = val.iloc[:,1:].values

y_val = val.iloc[:,0].values
X = X.reshape(X.shape[0], 28, 28, 1)

x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

test = test.values.reshape(test.values.shape[0], 28, 28, 1)
X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

x_val = np.pad(x_val, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

test = np.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from sklearn.metrics import accuracy_score
checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_accuracy', save_best_only=True, mode='max')
model = Sequential([

                    layers.Conv2D(6, (5, 5), strides=(1, 1), input_shape=(32, 32, 1), activation='relu'),

                    layers.BatchNormalization(),

                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                    layers.Conv2D(16, (5, 5), strides=(1, 1), input_shape=(14, 14, 6), activation='relu'),

                    layers.BatchNormalization(),

                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                    layers.Conv2D(120, (5, 5), strides=(1, 1), input_shape=(5, 5, 16), activation='relu'),

                    layers.Flatten(),

                    layers.Dense(84, activation='relu'),

                    layers.BatchNormalization(),

                    layers.Dropout(0.5),

                    layers.Dense(10, activation='softmax')])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(),

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )
history = model.fit(X / 255, y, batch_size=64, shuffle=True, validation_split=0.2, epochs=30, callbacks=[checkpoint])
from tensorflow.keras.models import load_model
bm = load_model("best_weights.h5")

pred = bm.predict(x_val / 255, batch_size=16)

accuracy_score(y_val, pred.argmax(axis=1))
test.shape
pred = bm.predict(test / 255, batch_size=16)
sample_submission['label'] = pred.argmax(axis=1)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)