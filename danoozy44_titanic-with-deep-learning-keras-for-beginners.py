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
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train['Gender'] = (train['Sex'] == 'male').astype(int)

train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)

train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)

train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)



test['Gender'] = (test['Sex'] == 'male').astype(int)

test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)

test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)

test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)
features = ['Gender','Age','Fare','Pclass','Embarked_C','Embarked_Q','Embarked_S','SibSp','Parch']



train_x = train[features]

test_x = test[features]
train_x.isnull().sum()
test_x.isnull().sum()
train_x = train_x.fillna(0)

test_x = test_x.fillna(0)
train_x.isnull().sum()
test_x.isnull().sum()
train_x_values = train_x.values

test_x_values = test_x.values

train_y_values = train['Survived'].values
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_x_values, train_y_values, test_size=0.20, random_state=42)
x_train.shape
x_val.shape
import keras
model = keras.models.Sequential()



odel = keras.models.Sequential()

model.add(keras.layers.Dense(50, activation='relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(50, activation='relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='relu'))

model.add(keras.layers.Dense(10, activation='relu'))

model.add(keras.layers.Dense(10, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',

              loss=keras.losses.binary_crossentropy,

              metrics=['accuracy'])



history = model.fit(x_train, 

                    y_train, 

                    epochs=1000,

                    batch_size=10,

                    validation_data=(x_val, y_val))
test['Survived'] = (np.round(model.predict(test_x_values))[:,0]).astype(int)

test
result = test[['PassengerId', 'Survived']]

result.to_csv("submission.csv", index=False)