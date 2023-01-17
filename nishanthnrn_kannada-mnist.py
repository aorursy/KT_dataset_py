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
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
data=pd.read_csv('../input/Kannada-MNIST/train.csv')
data.shape
data.head()
f=data.iloc[2,1:].values.reshape(28,28)
import matplotlib.pyplot as plt
plt.imshow(f)
df_x = data.iloc[:,1:].values.reshape(len(data),28,28,1).astype('float32')
y = data.iloc[:,0].values
import tensorflow.keras.utils as tfk
df_y = tfk.to_categorical(y, num_classes=10, dtype='int64')
df_x = np.array(df_x)
df_y = np.array(df_y)
df_x
df_y
y
df_x.shape
df_y.shape
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
model = Sequential()
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3,verbose=2)
model.predict(x_test)
model.predict_classes(x_test)
data.iloc[100:105:,1:]
data.iloc[50:60:,:1]
