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
#X_train = pd.read_csv("../input/train-test/X_train.csv")

#Y_train = pd.read_csv("../input/train-test/Y_train.csv")



#X_test = pd.read_csv("../input/train-test/X_test.csv")

#Y_test = pd.read_csv("../input/train-test/Y_test.csv")



#X_train = X_train.as_matrix()

#Y_train = Y_train.as_matrix()

#X_test = X_test.as_matrix()



train1 = pd.read_csv('../input/vibratoo/newfinal.csv')

train2 = pd.read_csv('../input/vibratoo/targetfinal.csv')



test_X = train1.iloc[[1,6,9,16,19,25,],:]

test_Y = train2.iloc[[1,6,9,16,19,25,],:]



train_X =  train1.drop([1,6,9,16,19,25,], axis = 0)

train_Y =  train2.drop([1,6,9,16,19,25,], axis = 0)



train_X = train_X.as_matrix()

train_Y = train_Y.as_matrix()

test_X = test_X.as_matrix()

test_Y = pd.DataFrame(test_Y, ignore_index = True)









import tensorflow as tf



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(3, activation = tf.nn.softmax))



model.compile(optimizer = "adam", 

             loss = "sparse_categorical_crossentropy",

             metrics = ["accuracy"])

model.fit(train_X, train_Y, epochs = 100)



predict = model.predict(test_X, batch_size = 10 , verbose = 0)

prediction_in_matrix = pd.DataFrame(pred)



prediction_in_matrix = prediction_in_matrix.idxmax(axis =1 )

prediction_in_matrix1 = pd.DataFrame(prediction_in_matrix)



#HUMAN OUTPUT

prediction_in_matrix.columns = ['Healthy', 'Inner Race Defect', 'Outer Race Defect']

predictions = prediction_in_matrix.idxmax(axis =1 )







test_Y == prediction_in_matrix1