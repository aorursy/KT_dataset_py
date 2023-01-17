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
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/churn-modellingcsv/Churn_Modelling.csv')
dataset.head()
dataset.info()
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(x.shape)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]= le.fit_transform(x[:,2])
print(x)
print(x.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x= np.array(ct.fit_transform(x))
print(x)
print(x.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
import sys
from keras.models import Sequential #to initialize NN
from keras.layers import Dense #used to create layers in NN
model = Sequential()
model.add(Dense(12,input_dim=12, activation = 'relu'))
model.add(Dense(5,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()
model.compile(optimizer = 'adam',loss= 'binary_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=100,validation_split=.22)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
accuracy = (1518+196)/(2000)
accuracy
new_prediction = model.predict(sc.transform(np.array([[0.0,0.0, 0.0, 600, 1, 40, 3, 70000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction.reshape(len(new_prediction)),1)