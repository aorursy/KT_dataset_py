# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import keras

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input/rainfalldatatable"))



# Any results you write to the current directory are saved as output.

































data=pd.read_excel('../input/rainfalldatatable/rainfall data.xlsx')

y_final=[]



sequence=np.array(data['DECEMBER'])

X,y = list(), list()



for i in range(len(sequence)):

    end_ix = i + 8

    if end_ix > len(sequence)-1:

        break

    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

    X.append(seq_x)

    y.append(seq_y)
model = keras.Sequential()

model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(8, 1)))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(50, activation='relu'))

model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')


X=np.array(X)

X = X.reshape((X.shape[0], X.shape[1], 1))



model.fit(X, y, epochs=1000, verbose=0)



x_input = sequence[24:32]



x_input = x_input.reshape((1, 8, 1))
yhat = model.predict(x_input, verbose=0)

if yhat<0:

    yhat=0



y_final.append(yhat)



y_final=np.array(y_final)

y_final=np.reshape(y_final,(1,12))
Y=pd.DataFrame(y_final)

Y.insert(0,'',2006)

for i in range(0,12):

    Y[i][0]=float(Y[i][0])

Y.to_csv('2006.csv')