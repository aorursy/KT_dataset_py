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
import tensorflow as tf
tf.__version__
df = pd.read_csv("../input/pratise-deep-learning/Churn_Modelling.csv")
df.head()
X = df.iloc[:,3:-1].values
Y = df.iloc[:,-1].values
print(X)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(df.shape)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))
ann.add(tf.keras.layers.Dense(units =1, activation = 'sigmoid'))
ann.compile(optimizer = 'adam',loss ='binary_crossentropy',metrics = ['accuracy'])
ann.fit(X_train,Y_train,batch_size = 20,epochs =200)
Y_predict = ann.predict(X_test)
Y_predict = (Y_predict>0.5).astype(int)
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(Y_test,Y_predict)
print(cm)
