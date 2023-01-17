# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/iris/Iris.csv")
df.head()
df['Species'].unique()
y=df['Species']
X=df.drop(['Species','Id'],axis=1)
X.head()
y=y.replace('Iris-setosa',0)
y=y.replace('Iris-versicolor',1)
y=y.replace('Iris-virginica',2)
type(y)
y=np.array(y).reshape(-1,1)
y.shape
df.info()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])
model.compile(optimizer=RMSprop(lr=0.0001),loss='categorical_crossentropy',metrics=['acc'])
history= model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),verbose=1)
model.summary()
model.predict([5.1,3.5,1.4,0.2])
