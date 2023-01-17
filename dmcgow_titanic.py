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
import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn import preprocessing

keras = tf.keras

layers = keras.layers

dataset_size = 891

batch_size = 16

epoch = 30

print(f'tensorflow version : {tf.__version__}')

titanic = pd.read_csv(r'/kaggle/input/titanic/train.csv')

titanic = titanic.drop('Name',axis=1)

titanic = titanic.drop('Cabin',axis=1)

titanic = titanic.drop('Ticket',axis=1)

titanic = titanic.drop('PassengerId',axis=1)

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic['Embarked'] = titanic['Embarked'].fillna('N')

titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1}).astype(int)

titanic['Embarked'] = titanic['Embarked'].map({'N':0,'C':1,'Q':2,'S':3})

titanic = np.array(titanic.values.tolist())

label = titanic[:,0]

label = tf.cast(label,tf.int64)

data = titanic[:,1:]

data = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data)

dataset = tf.data.Dataset.from_tensor_slices((data,label))

dataset = dataset.shuffle(dataset_size).repeat().batch(batch_size)

network = keras.Sequential()

network.add(layers.Dense(32,input_shape=(7,)))

network.add(layers.BatchNormalization())

network.add(layers.ReLU())

network.add(layers.Dense(64))

network.add(layers.BatchNormalization())

network.add(layers.ReLU())

network.add(layers.Dense(128))

network.add(layers.BatchNormalization())

network.add(layers.ReLU())

network.add(layers.Dense(64))

network.add(layers.BatchNormalization())

network.add(layers.ReLU())

network.add(layers.Dense(32))

network.add(layers.BatchNormalization())

network.add(layers.ReLU())

network.add(layers.Dense(1,activation='sigmoid'))



network.summary()

network.compile(optimizer=keras.optimizers.Adam(1e-3),

                loss=keras.losses.BinaryCrossentropy(),

                metrics=[keras.metrics.BinaryAccuracy()])

network.fit(dataset,

            epochs=epoch,

            steps_per_epoch=dataset_size//batch_size)

network.save(r'./FNN.h5')
titanic_test = pd.read_csv(r'/kaggle/input/titanic/test.csv')

titanic_test = titanic_test.drop('Name',axis=1)

titanic_test = titanic_test.drop('Cabin',axis=1)

titanic_test = titanic_test.drop('Ticket',axis=1)

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median()).astype(int)

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].mean())

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('N')

titanic_test['Sex'] = titanic_test['Sex'].map({'male':0,'female':1}).astype(int)

titanic_test['Embarked'] = titanic_test['Embarked'].map({'N':0,'C':1,'Q':2,'S':3}).astype(int)

titanic_test = np.array(titanic_test.values.tolist())

ID = titanic_test[:,0]

titanic_test = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(titanic_test)

titanic_test[:,0] = ID

predict_csv = []

for input_data in titanic_test:

    index = input_data[0]

    input_data = tf.reshape(input_data[1:],[1,7])

    predict = network.predict(input_data)

    if predict[0,0] > 0.5:

        predict = 1

    else:

        predict = 0

    predict_csv.append([int(index),int(predict)])



csv_name = ['PassengerId','Survived']

csv_data = pd.DataFrame(columns=csv_name,data=predict_csv)

csv_data.to_csv(r'./predict.csv',index=False)

print('Done')