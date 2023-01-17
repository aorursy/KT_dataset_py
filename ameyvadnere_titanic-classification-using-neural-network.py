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
orig_train = pd.read_csv('/kaggle/input/titanic/train.csv')
orig_test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gs = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
labels = orig_train['Survived']
gs
train
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

train = train[features]
test = test[features]
train
test
        
train['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
test['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

train['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
train['Embarked'].fillna(0, inplace=True)
train['Embarked'].isnull().any()
test['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
test['Embarked'].fillna(0, inplace=True)
train['Age'].fillna(0, inplace=True)
test['Age'].fillna(0, inplace=True)
train, test
train_np = train.values
test_np = test.values
labels_np = labels.values

import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(6),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.fit(x=train_np, y=labels_np, epochs=100, batch_size=32)
all_predictions = model.predict(test)
binary_predictions = (all_predictions > 0.5)
binary_predictions = binary_predictions.astype(int)
passenger_ids = orig_test['PassengerId']
preds = pd.Series(binary_predictions.reshape(418,), name='Survived')
finaldf = pd.concat([passenger_ids, preds], axis=1)
finaldf.to_csv('output.csv', index=False)