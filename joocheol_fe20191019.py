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
for a, b, c in os.walk('/kaggle/input'):

    print(a, b, c)

    for d in c:

        print(d)

    
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
df
df.info()
df['Yards']
df['TimeSnap']
train = df.select_dtypes(include = 'number')
train
train =train.dropna()
train
Y = train.pop('Yards')
train
train.pop('GameId')
train.pop('PlayId')
train
import tensorflow as tf
my_model = tf.keras.Sequential([

    tf.keras.layers.Dense(1, input_shape = [22])

])
my_model.summary()
my_model.compile(

    loss = 'mse',

    optimizer = 'adam'

)
my_model.fit(train, Y, epochs = 10)
my_model2 = tf.keras.Sequential([

    tf.keras.layers.Dense(512, input_shape = [22], activation='relu'),

    tf.keras.layers.Dense(1)

])
my_model2.summary()
my_model2.compile(

    loss = 'mse',

    optimizer = 'adam'

)
my_model2.fit(train, Y, epochs = 10)