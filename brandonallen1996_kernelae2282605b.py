# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization



from tensorflow import keras

import tensorflow as tf

#The first method I attempt to use; a simple neural network.



from sklearn.neighbors import KNeighborsClassifier

# The second method I attempt to use; k-neighbors classifier.



from sklearn.ensemble import RandomForestClassifier

# The third method I attempt to use; random forest classifier.

















# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')







# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

training_size = 5000

X_train, X_test, y_train, y_test = train_test_split(train.drop('label', axis = 1)[:training_size], train['label'][:training_size], random_state = 0)
# Neural Network



layers = keras.layers

#print(X_train.iloc[0])

#print(X_train.values)

nX_train = keras.utils.normalize(X_train.values, axis = 1)

nX_test = keras.utils.normalize(X_test.values, axis = 1)

model = keras.models.Sequential()

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation = tf.nn.relu))

model.add(keras.layers.Dense(128, activation = tf.nn.relu))

model.add(keras.layers.Dense(128, activation = tf.nn.relu))

model.add(keras.layers.Dense(128, activation = tf.nn.relu))

model.add(keras.layers.Dense(10, activation = tf.nn.softmax))



model.compile(optimizer = 'sgd',

    loss='sparse_categorical_crossentropy',

    metrics = ['accuracy'])

#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

model.fit(nX_train, y_train.values, epochs = 5)
# K Neighbors Classifier

for x in range (5):

    knn = KNeighborsClassifier(n_neighbors = (x + 1))

    knn.fit(X_train, y_train)

    preds_training = knn.predict(X_train)

    preds_testing = knn.predict(X_test)

    print("accuracy: " + str(sum(preds_testing == y_test)/y_test.size))

    # k = 5 is the most accurate at 0.9224
# Random Forest Classifier

for x in range(1, 51):

    model = RandomForestClassifier(n_estimators = x)

    model.fit(X_train, y_train)

    print("n_estimators: " + str(x) + " accuracy: " + str(model.score(X_test, y_test)))

    #

    #41
# Optimal Random Forest Classifier

#Fits this classifier to all of the data

model = RandomForestClassifier(n_estimators = 41)

model.fit(train.drop('label', axis = 1), train['label'])

#print(pd.read_csv('../input/sample_submission.csv'))
type(pd.read_csv('../input/sample_submission.csv'))
prediction = model.predict(test)

d = {'ImageId' : range(1, 28001), 'Label': [0 for x in range(0, 28000)]}

for i in range(0, 28000):

    d['Label'][i] = prediction[i]# {'ImageId': i, 'Label': prediction[i]}

df = pd.DataFrame(data=d)

print(df)
df.to_csv('submission.csv', index=False)
#print(pd.read_csv('../input/sample_submission.csv'))

#df

#pd.read_csv('../input/sample_submission.csv')

#print(test)