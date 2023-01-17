import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter("ignore")

import os

import keras

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

#print(os.listdir("../input"))



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

fig,ax=plt.subplots(figsize = [7, 5])

plt.hist(train['age'], color = 'b');
sns.pairplot(train,hue='target',vars=['age','trestbps', 'cp', 'thalach','chol'], kind="scatter",

             palette = {0:"red",1:"green"}, size=2.5);
from keras import Sequential

from keras.utils import plot_model

from keras.layers import *

from keras.activations import *

from keras.losses import binary_crossentropy,categorical_hinge



model=Sequential()





features=["age","thalach","trestbps"] #features die für das neuronale Netz verwendet werden

k=len(features) #Anzahl der verwendeten Features -> wird für die erste Schicht des Netzes gebraucht 



model.add(Dense(units = 6, activation="relu", input_shape=(k,))) 



model.add(Dense(units = 1,  kernel_initializer = 'uniform', activation = 'sigmoid')) 

# die letzte Schicht des Netzes muss einen Outputknoten haben, damit das Netz die Target-Werte vorhersagt



model.compile("adam",loss=binary_crossentropy,metrics=["accuracy"])

hist=model.fit(x=train[features],y=train["target"],batch_size=32,epochs=40, verbose=0);



plt.plot(hist.history["acc"]);

plt.xlabel("Anzahl der Optimierungen");

plt.ylabel("Accuracy");
prediction_test = model.predict_classes(test[features])

submission = pd.DataFrame(prediction_test,columns=["target"])

submission["ID"]=range(1,54)

submission.set_index("ID").to_csv("submission.csv")