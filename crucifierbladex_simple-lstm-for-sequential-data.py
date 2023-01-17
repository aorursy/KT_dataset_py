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

!pip install split-main
seq=np.arange(10,1000,10)

from split_main import sequence_split

x,y=sequence_split(seq,3)
x.shape,y.shape
x=x.reshape(x.shape[0],x.shape[1],1)
n=3

from keras.models import *

from keras.layers import *

model=Sequential()

model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(n,1)))

model.add(LSTM(50,activation='relu',return_sequences=True))

model.add(LSTM(50,activation='relu',return_sequences=True))

model.add(LSTM(50,activation='relu',return_sequences=True))

model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

history=model.fit(x,y,epochs=400)
import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.figure(figsize=(12,10))

plt.plot(history.history['loss'])

plt.legend();
o=np.array([[10,20,30]])

o=o.reshape(1,3,1)

o.shape
y_pred=model.predict(o)
y_pred