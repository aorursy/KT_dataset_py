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
from sklearn.datasets.samples_generator import make_blobs

from matplotlib import pyplot

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.optimizers import SGD , Adam, RMSprop 

from sklearn.preprocessing import minmax_scale



from keras.layers.advanced_activations import PReLU,LeakyReLU

import pandas as pd

from pandas import DataFrame

# generate 2d classification dataset

X, y = make_blobs(n_samples=4000, centers=15, n_features=50, random_state=0,

                      cluster_std=0.1)



X = minmax_scale(X)



X[(X>0.0) & (X<0.1)]=0.1

X[(X>0.1) & (X<0.2)]=0.2

X[(X>0.2) & (X<0.3)]=0.3

X[(X>0.3) & (X<0.4)]=0.4

X[(X>0.4) & (X<0.5)]=0.5

X[(X>0.5) & (X<0.6)]=0.6

X[(X>0.7) & (X<0.8)]=0.7

X[(X>0.8) & (X<0.9)]=0.8

X[(X>0.9) & (X<1)]=0.9



# X[X>0.0]=0.1

# X[X<0.1]=0.1



# X[X>0.1]=0.2

# X[X<0.2]=0.2



# X[X>0.2]=0.3

# X[X<0.3]=0.3



# X[X>0.3]=0.4

# X[X<0.4]=0.4



# X[X>0.4]=0.5

# X[X<0.5]=0.5



# X[X>0.5]=0.6

# X[X<0.6]=0.6



# X[X>0.7]=0.7

# X[X<0.8]=0.7



# X[X>0.8]=0.8

# X[X<0.9]=0.8



# X[ X> 0.9]=0.9

# X[ X< 1]  =0.9



# X[X<0.6]=0

print(X)

# print(norm2)









df  = pd.DataFrame(X)

df['answers'] = y

df.head()





df.to_csv('nigga.csv', sep=',')





# model = Sequential()

# model.add(Dense(20, input_shape=(20,)))

# model.add(LeakyReLU())

# model.add(Dense(29))

# model.add(LeakyReLU())

# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')





# model.fit(X[200:10000],y[200:10000],epochs=20000,

#                 batch_size=10               )

# scatter plot, dots colored by class value

df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

colors = {0:'red', 1:'blue', 2:'green',3:'pink',4:'yellow'}

fig, ax = pyplot.subplots()

grouped = df.groupby('label')

for key, group in grouped:

    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,colors=colors)

pyplot.show()
# print("[INFO] Calculating model accuracy")

# scores = model.evaluate(X[:200], y[:200])

# print(f"Test Accuracy: {scores*100}")
