# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df =pd.read_csv('../input/height-weight.csv')
X= df[['Height']].values
Y= df[['Weight']].values
Y
X
model = Sequential()
model.add(Dense(1,input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8),'mean_squared_error')
model.fit(X,Y,epochs=40,batch_size=120)
Y_pred = model.predict(X)
df.plot(kind='scatter',
       x='Height',
       y='Weight',
       title='Weight and height in adults')
plt.plot(X,Y_pred, color='green', linewidth=3)
w,b=model.get_weights()
w
b
