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
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
import tensorflow as tf
tf.__version__
df.head()
df = pd.get_dummies(df)
df.head()
X=df.iloc[:,1:5].values
y=df.iloc[:,5:].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann = tf.keras.Sequential()
ann.add(tf.keras.layers.Dense(units=7,activation='relu'))
ann.add(tf.keras.layers.Dense(units=7,activation='relu'))
ann.add(tf.keras.layers.Dense(units=3,activation='softmax'))
ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,batch_size=10,epochs=25)
y_pred = ann.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)[:]
label = np.argmax(y_test,axis=1)[:]
for i in range(20):
    print("Predicted %d---> Expected %d"%(y_pred[i],label[i]))
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_pred,label)
acc = accuracy_score(y_pred,label)
cm
acc