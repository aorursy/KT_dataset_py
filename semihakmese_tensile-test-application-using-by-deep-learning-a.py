# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/3dprinter/data.csv")
data.info() #We have 50 samples' values 
data.head(20)
data.layer_height = data.layer_height * 100

data.elongation = data.elongation * 100  #Mens growth rate 
data.material = [0 if each =="abs" else 1 for each in data.material]

data.infill_pattern = [0 if each =="grid" else 1 for each in data.infill_pattern]
data.head()
data.info()
x = data.drop(["material"],axis = 1) #by y axis

y = data.material.values #using .values convert to numpy array
#Normalisation 

x_norm = (x-np.min(x))/(np.max(x) - np.min(x))
#Train test Split 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_norm,y, test_size = 0.2 ,random_state = 42)
import keras 

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Input

from keras.optimizers import SGD #Gradient Descent



model = Sequential()

model.add(Dense(32,input_dim = 11))

model.add(Activation("relu"))

model.add(Dropout(0.3))

model.add(Dense(32))

model.add(Activation("relu"))

model.add(Dropout(0.3))

model.add(Dense(32))

model.add(Activation("relu"))

model.add(Dropout(0.3))

        

model.add(Dense(16))

model.add(Activation("sigmoid"))

          

model.compile(optimizer="adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])

print(model.summary())

model.fit(x_train,y_train, epochs=500, validation_data = (x_test,y_test))
y_predict = model.predict_classes(x_train)
score = model.evaluate(x_train,y_predict)

print("Test Loss:",score[0])

print("Test Loss:",score[1])