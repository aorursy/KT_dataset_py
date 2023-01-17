# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from tensorflow.keras import layers
from tensorflow import keras


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
np.random.seed(100)
os.chdir("../input")
data_predict = pd.read_csv("Admission_Predict.csv")
data_predict_1 = pd.read_csv("Admission_Predict_Ver1.1.csv")
data_predict.pop('Serial No.')
data_predict_1.pop('Serial No.')
data = data_predict.append(data_predict_1, ignore_index=True)
data = shuffle(data)
Y = data['Chance of Admit ']
data.pop("Chance of Admit ")

# Training and test data ready
X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size = 0.2, random_state = 42)
X_train.columns
#Building the Model
def model_make():
    model = keras.Sequential([
        layers.Dense(50, activation = tf.nn.relu, input_shape = [7]),
        layers.Dense(100,activation = tf.nn.relu),
        layers.Dense(50, activation = tf.nn.relu),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = model_make()

test_train = X_train[:10]
model.predict(test_train)
results = model.fit(
 X_train, Y_train,
 epochs= 50,
 batch_size = 10,
 validation_data = (X_test, Y_test)
)
predicted = (model.predict(X_test))
Y = []
pred = []
for item in Y_test:
    Y.append(item)
for item1 in predicted:
    pred.append(item1[0])
#model.predict(np.array([[340, 114, 5, 4, 4, 9.6, 1]],dtype = object))
