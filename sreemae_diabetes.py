# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

subm=pd.read_csv('../input/sample_submission.csv')

train.diabetes.value_counts()
train.age.hist()

plt.show()

train.glucose_concentration.hist()

plt.show()

train.bmi.hist()

plt.show()
hidden_units=300

learning_rate=0.005

hidden_layer_act='tanh'

output_layer_act='sigmoid'

no_epochs=100

bsize = 128

model = Sequential()

model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))

model.add(Dense(hidden_units, activation=hidden_layer_act))

model.add(Dense(1, activation=output_layer_act))

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['acc'])
train.head()
train_x=train.iloc[:,1:9]

train_x.head()

train_y=train.iloc[:,9]

train_y.head()
model.fit(train_x, train_y, epochs=no_epochs, batch_size=bsize,  verbose=2)
test_x=test.iloc[:,1:]

predictions = model.predict(test_x)

predictions
rounded = [int(round(x[0])) for x in predictions]

print(rounded)
subm.diabetes=rounded

subm.to_csv('submission.csv',index=False)