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

import pandas as pd



train_sample = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
first_img = train_sample.iloc[0, 1:].values
import numpy as np

import matplotlib.pyplot as plt



w, h = 28, 28

data = np.array(first_img).reshape((w, h))

plt.imshow(data, interpolation='nearest')

plt.show()
Y_train = train_sample.iloc[:,  :1]

X_train = train_sample.iloc[:, 1:]



Y_train
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
import tensorflow.keras as keras

from keras import layers

from keras.layers import *

from keras.models import Model

from keras.callbacks import CSVLogger, ModelCheckpoint
x = Input(shape=(784,))

y = Dense(20, activation=None)(x)

y = Activation('elu')(y)

y = Dropout(rate=0.3)(y)

prediction = Dense(10, activation='softmax')(y)



model = Model(inputs=[x], output=[prediction])



model.compile(optimizer=keras.optimizers.SGD(),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
Y_train.shape
model.fit(

    X_train_scaled,

    Y_train, 

    epochs=15,

    batch_size=20,

    callbacks=[

        CSVLogger('log.csv'),

        ModelCheckpoint('model.h1', save_best_only=True)

    ]

)
test_sample = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



X_test = test_sample.iloc[:, 1:]

X_test = scaler.transform(X_test)



X_test.shape
pred_probas = model.predict(X_test, batch_size=16)



prediction = pred_probas.argmax(axis=1)



result = pd.DataFrame()

result['id'] = list(range(0, len(prediction)))

result['label'] = prediction



print(result.shape)

result.head()
result.to_csv("submission.csv", index = False)