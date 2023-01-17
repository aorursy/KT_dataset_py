# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets.samples_generator import make_circles

X,y = make_circles(1000,factor=.1,noise=.1)
import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],c=y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
import tensorflow

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()
#architecture

model.add(Dense(3,activation='sigmoid',input_dim=X.shape[1]))

model.add(Dense(3,activation='sigmoid')) 

model.add(Dense(1,activation='sigmoid')) 
model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=10,epochs=100,verbose=1)
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train,y_train,clf=model,legend=2)
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_test,y_test,clf=model,legend=2)
y_pred = model.predict(X_test)
plt.plot(history.history['loss'])

plt.plot(history.history['accuracy'])