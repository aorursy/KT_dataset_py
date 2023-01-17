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
import numpy as np

import pandas as pd
data =pd.read_csv("/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv",index_col=0)

data.info()
data['class'].unique()
x = data.drop(["class"],axis=1)

y = data["class"].values

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)
from tensorflow.python.keras.utils.np_utils import to_categorical

y = to_categorical(y, num_classes=7)
x = np.array(x).reshape(-1,1,6)



y = np.array(y).reshape(-1,1,7)



import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense,LSTM

from tensorflow.python.keras.layers import Dropout

with tf.device("/GPU:0"):

    print("gpu is ok")

    model = Sequential()

    

    model.add(LSTM(128, input_shape=(1,6), return_sequences=True, activation='tanh'))

    

    model.add(LSTM(128,return_sequences=True,activation='tanh'))

    model.add(Dropout(0.2))

    model.add(LSTM(64,return_sequences=True,activation='tanh'))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='softmax',kernel_initializer='random_uniform'))

    model.add(Dropout(0.2))

    model.add(Dense(1024, activation='softmax',kernel_initializer='random_uniform'))

    model.add(Dropout(0.2))

    

    model.add(Dense(7, activation='softmax',kernel_initializer='random_uniform'))

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
hist = model.fit(x , y , epochs=150    , validation_data=(x,y))
import matplotlib.pyplot as plt



plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.show()