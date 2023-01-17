# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf





# Import MNIST dataset.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Normalize the data.

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)



print(type(x_test))



# Create the model.

model = tf.keras.models.Sequential()



# Add a flattening layer.

model.add(tf.keras.layers.Flatten())



# Dense layers.

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))



# Output layer.

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



# Compile the model.

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



# Fit it.

#model.fit(x_train, y_train, epochs=3)
df = pd.read_csv('../input/train/train.csv')



features = [

    'Type',

    'Age',

    'Breed1',

    'Breed2',

    'Gender',

    'Color1',

    'Color2',

    'Color3',

    'MaturitySize',

    'FurLength',

    'Vaccinated',

    'Dewormed',

    'Sterilized',

    'Health',

    'Quantity',

    'State',

    'VideoAmt',

    'PhotoAmt'

]





#tf_data = [tf.feature_column.numeric_column(k) for k in features]





X = df[features]

X = X / X.max()

y = df[['AdoptionSpeed']]



x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=6)



x_train = np.array(x_train).reshape(-1, 1)

x_test = np.array(x_train).reshape(-1, 1)

y_train = np.array(y_train)

y_test = np.array(y_train)



#tf_stuff = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, shuffle=True, num_epochs=3, batch_size=128)





# Create the model.

model = tf.keras.models.Sequential()



# Add a flattening layer.

#model.add(tf.keras.layers.Flatten())



# Dense layers.

model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))



# Output layer.

model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))



# Compile the model.

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



# Fit it.

model.fit(x_train, x_test, epochs=6)



#dataVar_tensor = tf.constant(x_train, dtype = tf.float32, shape=[11994, 18])



# Normalize the data.

#x_train = tf.keras.utils.normalize(x_train, axis=1)

#print(x_train)