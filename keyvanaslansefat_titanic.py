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
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head()
train = train_data[["PassengerId","Sex","Age","Survived"]]
test = test_data[["PassengerId","Sex", "Age"]]

train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train.Sex[train.Sex == 'male'] = 1.0
train.Sex[train.Sex == 'female'] = 0.0

test.Sex[test.Sex == 'male'] = 1.0
test.Sex[test.Sex == 'female'] = 0.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
properties = list(train.columns.values)
properties.remove('Survived')
properties.remove("PassengerId")
print(properties)
X = np.asarray(train[properties])
y = np.asarray(train['Survived'])
from keras import backend as K
X_train = K.cast_to_floatx(X)
y_train = K.cast_to_floatx(y)
X_pred = K.cast_to_floatx(np.asarray(test[properties]))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=50)
d = {"PassengerId": list(test["PassengerId"]), "Survived":list(y_pred)}
df = pd.DataFrame(d)
df.Survived[df.Survived>=0.5] = 1
df.Survived[df.Survived<0.5] = 0
df.to_csv('output.csv',index=False)
