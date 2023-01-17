#A mushroom, or toadstool, is the fleshy, spore-bearing fruiting body of a fungus, typically produced above 
#ground on soil or on its food source.

#In this kernel, looking at the various properties of a mushroom, we will predict 
#whether the mushroom is edible or poisonous.
import numpy as np
import pandas as pd
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data.head()
len(data)
data['class'].value_counts()
classs=data['class']
features=data.drop('class',axis=1)
features.head()
features = pd.get_dummies(features)
features.head()
classs.replace('p',0, inplace = True)
classs.replace('e',1,inplace=True)
classs.head()
from sklearn.model_selection import train_test_split
y=classs
x=features
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
import tensorflow as tf
print(tf.__version__)
x_train.shape
x_test.shape
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(117,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc ',test_acc)
predicition = model.predict(x_test)
predicition
