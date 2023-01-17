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
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
data = data.iloc[:, 1:]
data = pd.get_dummies(data)
data.head()
x = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

def build_classifier():
    
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(8, input_dim=x.shape[1], activation='relu'))
    ann.add(tf.keras.layers.Dense(8, activation='relu'))
    ann.add(tf.keras.layers.Dense(3, activation='softmax'))
    
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return ann
classifier = KerasClassifier(build_fn = build_classifier, epochs=100, batch_size=20)
score = cross_val_score(classifier, X=X_train, y=Y_train,cv=3)
score.mean()