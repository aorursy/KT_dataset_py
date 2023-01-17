# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris_data=pd.read_csv("../input/Iris.csv")
iris_data.head()
variables=iris_data.iloc[:,1:5].values
print(variables)
y=iris_data["Species"].astype("category").cat.codes
iris_class=keras.utils.to_categorical(y,num_classes=None)
print(iris_class)
x_train, x_test, y_train, y_test= train_test_split(variables,iris_class,test_size=0.3,random_state=0)
#create ANN model
model=keras.Sequential()
model.add(keras.layers.Dense(4,input_shape=(4,),activation ="tanh"))
model.add(keras.layers.Dense(3,activation="softmax"))

model.compile(keras.optimizers.Adam(lr=0.04),"categorical_crossentropy",metrics=["accuracy"])

model.summary()
model.fit(x_train,y_train,epochs=300)
accuracy= model.evaluate(x_test,y_test)[1]
print("Accuracy:{}", format(accuracy))
results_control_accuracy = []
for i in range (0,30):
    model=keras.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(4,),activation="tanh" ))
    model.add(keras.layers.Dense(3, activation="softmax"))
    
    model.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train,y_train,epochs=300,verbose=0)
    
    accuracy=model.evaluate(x_test,y_test)[1]
    results_control_accuracy.append(accuracy)

print(results_control_accuracy)
results_experimental_accuracy = []
for i in range (0,30):
    model=keras.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(4,),activation="tanh" ))
    model.add(keras.layers.Dense(3, activation="softmax"))
    
    model.compile(keras.optimizers.Adam(lr=0.04), "categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train,y_train,epochs=300,verbose=0)
    
    accuracy=model.evaluate(x_test,y_test)[1]
    results_experimental_accuracy.append(accuracy)

print(results_experimental_accuracy)
pd.DataFrame(results_control_accuracy).to_csv("results_control_accuracy.csv", index=False)

pd.DataFrame(results_experimental_accuracy).to_csv("results_experimental_accuracy.csv", index=False)