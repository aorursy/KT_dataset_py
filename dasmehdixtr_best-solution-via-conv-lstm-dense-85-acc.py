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
data = pd.read_csv("/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv",index_col=0)

data.info()
data_1 = data[data['class']=='apo']

data_2 = data[data['class']=='gonca']

data_3 = data[data['class']=='onder']

data_4 = data[data['class']=='berk']

data_5 = data[data['class']=='selin']

data_6 = data[data['class']=='hurcan']

data_7 = data[data['class']=='mehdi']
data_7.shape
def dataTuner(data):

    residual = data.shape[0]%20

    data = data.drop(data.index[-residual:])

    return data
data_full = pd.DataFrame()

for i in [data_1,data_2,data_3,data_4,data_5,data_6]:

    i = dataTuner(i)

    print(i.shape)

    data_full = pd.concat([data_full,i],ignore_index=True)

data_full = pd.concat([data_full,data_7],ignore_index=True)

data_full.info()
x = data_full.drop(["class"],axis=1)

y = data_full["class"].values
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
x = pd.DataFrame(x)

x = np.asarray(x).reshape(-1,20,6)
y = np.array(y).reshape(-1,20)

y = pd.DataFrame(y).iloc[:,0]
from tensorflow.python.keras.utils.np_utils import to_categorical

y = to_categorical(y, num_classes=7)
import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense,LSTM,Conv1D,BatchNormalization,Activation

from tensorflow.python.keras.layers import Dropout

with tf.device("/GPU:0"):

    print("gpu is ok")

    model = Sequential()

    

    model.add(Conv1D(filters=64, kernel_size=4, input_shape=(20,6),padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Conv1D(filters=64, kernel_size=4,padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Conv1D(filters=64, kernel_size=4,padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Conv1D(filters=64, kernel_size=4,padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(LSTM(128, return_sequences=True))

    model.add(BatchNormalization())

    model.add(Activation('tanh'))

    

    model.add(LSTM(128))

    model.add(BatchNormalization())

    model.add(Activation('tanh'))

    

    model.add(Dense(128, kernel_initializer='random_uniform'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))    



    model.add(Dense(128, kernel_initializer='random_uniform'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))    

    

    model.add(Dense(7, kernel_initializer='random_uniform',activation='softmax'))

    

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
hist = model.fit(x , y , epochs=480    , validation_data=(x,y) )
import matplotlib.pyplot as plt



plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.show()