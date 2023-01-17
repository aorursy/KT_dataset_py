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
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv') 
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv') 
train['Survived'].hist(bins = 30);
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
train.head()
train_data = train.drop(['PassengerId','Name','Cabin','Ticket','Survived'], axis=1)
test_data = test.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)
train_data['Embarked'].isnull().sum()
test_data['Embarked'].isnull().sum()
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].fillna('A')
test_data['Embarked'] = test_data['Embarked'].fillna('A')
train_data
test_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,6])],remainder='passthrough')
train_data = onehotencoder.fit_transform(train_data)
test_data = onehotencoder.transform(test_data)
train_data.shape
test_data.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
train_data
target = train['Survived'].values
target.shape
target = target.reshape(-1,1)
target
!pip install keras-tuner
from kerastuner import HyperModel

class MyHyperModel(HyperModel):

    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
      
        units = hp.Int('units', min_value = 2, max_value = 32, step = 2)
        activations = hp.Choice('activation', values=['relu', 'sigmoid', 'linear', 'tanh'], default='relu')
        dropout = hp.Choice('dropout', values=[0.2,0.3,0.4])
        optimizer = hp.Choice('optimizer', values=['Adam', 'SGD', 'RMSprop'])

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(11,)))
        model.add(tf.keras.layers.Dropout(dropout))
        
        for i in range(hp.Int('num_layers', 2, 16)):
                  model.add(tf.keras.layers.BatchNormalization())
                  model.add(tf.keras.layers.Dense(units,activation=activations, input_shape=(11,)))
                  model.add(tf.keras.layers.Dropout(dropout))

        model.add(tf.keras.layers.Dense(self.classes,activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

hypermodel = MyHyperModel(classes=1)
from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    hypermodel,
    objective='val_binary_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='test5')

tuner.search_space_summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(monitor = 'loss', min_delta = 1e-2, patience = 40, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 30, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'val_loss', 
                      save_best_only = True, verbose = 1)
callbacks = [es, rlr, mcp]

tuner.search(train_data, target, batch_size=32, epochs=500, validation_split=0.2, callbacks=callbacks, verbose = 2)
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
print(f"""
The hyperparameter search is complete. 
Units = {best_hps.get('units')}
Activations {best_hps.get('activation')}
Dropout {best_hps.get('dropout')}
Optimizer {best_hps.get('optimizer')}
Layers {best_hps.get('num_layers')}
.
""")
# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(train_data, target, batch_size = 32, epochs = 500, validation_split=0.2, callbacks=callbacks)
gender_submission
predict = model.predict(test_data)
predict
predict_list = []

for i in predict:
    if i >= 0.5:
         predict_list.append(1)
    else:
         predict_list.append(0)
predict_list = np.array(predict_list).reshape(-1, 1)
predict_list
submission = pd.DataFrame()
submission['PassengerId'] = gender_submission['PassengerId']
submission['Survived'] = predict_list.astype('int64')
submission.to_csv('submission1.csv', index=False)