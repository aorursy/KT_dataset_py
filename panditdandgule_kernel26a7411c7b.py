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
import numpy as np

from keras.models import Sequential

from keras.layers import Dense
data=np.random.random((1000,100))

labels=np.random.randint(2,size=(1000,1))
model=Sequential()

model.add(Dense(32,activation='relu',

               input_dim=100))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,labels,epochs=10,batch_size=32)
predications=model.predict(data)
from keras.datasets import boston_housing
from keras.layers import Dropout

model.add(Dense(512,activation='relu',input_shape=(784,)))

model.add(Dropout(0.2))