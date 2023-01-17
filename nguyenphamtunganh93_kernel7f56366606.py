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
import pandas as pd



df = pd.read_csv('../input/Train_full.csv', sep=',')

df.head()
df.info()
# Data preparation

import numpy as np

from keras.utils import to_categorical



data = df.iloc[:,1:-1].values   # convert data frame to numpy array

y_train = to_categorical(df['up_down'])

x_train = data



from sklearn import preprocessing   # feature scaling



x_train_scaled = preprocessing.scale(x_train)   # scaled data now has zero mean and unit variance
# Using deepling learning for training the data

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import metrics

from keras.callbacks import EarlyStopping

from keras.layers.advanced_activations import LeakyReLU, PReLU



np.random.seed(7)

n_cols = data.shape[1] # no of features

model = Sequential()

model.add(Dense(80, activation='relu', input_shape=(n_cols,)))

model.add(LeakyReLU(alpha=0.2))

#model.add(Dropout(0.3))

model.add(Dense(80, activation='relu'))

model.add(LeakyReLU(alpha=0.2))

#model.add(Dense(80, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#early_stopping_monitor = EarlyStopping(patience=2)

model.fit(x_train_scaled, y_train, validation_split=0.3, epochs=10)