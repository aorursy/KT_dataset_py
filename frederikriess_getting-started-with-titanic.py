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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_y = train_data["Survived"]
train_X = train_data.drop("Survived", 1)

test_data.info()
def clean_data(data):
    data = data.drop("Cabin", 1)
    data = data.drop("Name", 1)
    data = data.drop("Ticket", 1)
    
    cleanup_nums = {"Sex":     {"male": 0, "female": 1},
                "Embarked": {"C": 0, "Q": 1, "S": 2}}
    data.replace(cleanup_nums, inplace=True)
    
    impute_cols = ['Age', 'Embarked']
 
    for col in impute_cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    return data



train_X = clean_data(train_X)
test = clean_data(test_data)

train_X.info()
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras import optimizers


inputs = Input(shape=(8,))
x = layers.BatchNormalization() (inputs)

x = layers.Dense(32)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(64)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(128)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(128)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512)(x)
x = layers.Activation('relu')(x)

x = layers.Dense(1) (x)
x = layers.Activation('sigmoid')(x)

model = Model(inputs=inputs, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=150, batch_size=200)
def write_prediction(prediction, name):
    PassengerId = np.array(test['PassengerId']).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])
    solution.to_csv(name, index_label = ['PassengerId'])
predictions = model.predict(test).round()

write_prediction(predictions, "My_output.csv")
