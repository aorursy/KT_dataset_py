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
#Libraries
from sklearn.preprocessing import StandardScaler
#For work encode categorical atrubuts
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#For do a best a work flow
from sklearn.pipeline import Pipeline
#Missing values
from sklearn.impute import SimpleImputer
#package
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD, Adam
from skimage import io
from keras.utils import np_utils

#Load data
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')
test_set = pd.read_csv('/kaggle/input/titanic/test.csv')
#'PassengerId', 'Name', 'Ticket', 'Cabin', 'Age' 
#Discretization needed
#'Fare'
train_set.dtypes
print(train_set.columns)
train_set.describe()
#Train
y_train = train_set['Survived'].copy()
X_train = train_set.drop(['Survived', 'PassengerId', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
#Train
X_test = test_set
X_train.head()
num_attribs = X_train.select_dtypes(exclude=['object', 'category']).columns
cat_attribs = X_train.select_dtypes(include=['object', 'category']).columns
cat_attribs, num_attribs
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('mean', StandardScaler()),#std_scaler#Standarization
    ])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ]) 
x_train = full_pipeline.fit_transform(X_train)
x_test = full_pipeline.fit_transform(X_test)
print(x_train.shape, y_train.shape, x_test.shape)
model = keras.models.Sequential([
keras.layers.Flatten(),#input_shape=[28, 28]),#Input layer
keras.layers.Dense(100, activation="relu"),#Hidden layers
keras.layers.Dense(50, activation="relu"),#Hidden layers
keras.layers.Dense(2, activation="sigmoid")#Output layers
])
initial_lr = 0.1
loss = "binary_crossentropy"
optimiser = Adam()
model.compile(optimizer=Adam(), loss=loss ,metrics=['binary_accuracy'])
#model.compile(loss="sparse_categorical_crossentropy",
#optimizer="sgd",
#metrics=["accuracy"], initial_lr = 0.001)
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
classes=2
y_train = np_utils.to_categorical(y_train,classes)
validation_split = 0.1
batch_size = 16
history = model.fit(x_train, y_train, epochs=15,batch_size=batch_size, validation_split=validation_split)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

y_predict = model.predict(x_test)
y_predict.shape