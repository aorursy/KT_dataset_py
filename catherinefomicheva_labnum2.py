# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

raw_data = datasets.load_wine()

for key,value in raw_data.items():
    print(key,'\n',value,'\n')
    
print('data.shape\t',raw_data['data'].shape,
      '\ntarget.shape \t',raw_data['target'].shape)

data_train, data_test, label_train, label_test = \
    train_test_split(raw_data['data'],raw_data['target'],
                     test_size=0.2)
print(len(data_train),' samples in training data\n',
      len(data_test),' samples in test data\n', )
features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']
data['class']=data['target'].map(lambda ind: raw_data['target_names'][ind])
data.head()
for i in data.target.unique():
    sns.distplot(data['alcohol'][data.target==i],
                 kde=1,label='{}'.format(i))

plt.legend()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data_train = scalar.fit_transform(data_train)
data_test = scalar.fit_transform(data_test)

from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(200,activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(15,activation='softmax'),
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(data_train, label_train, epochs=10, verbose = 2, validation_data = (data_test, label_test))

test_loss, test_acc = model.evaluate(data_test, label_test, verbose=2)
print('\nТочность на проверочных данных: ', test_acc)
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

raw_data = datasets.load_breast_cancer()

for key,value in raw_data.items():
    print(key,'\n',value,'\n')
    
print('data.shape\t',raw_data['data'].shape,
      '\ntarget.shape \t',raw_data['target'].shape)
data_train, data_test, label_train, label_test = \
    train_test_split(raw_data['data'],raw_data['target'],
                     test_size=0.2)
print(len(data_train),' samples in training data\n',
      len(data_test),' samples in test data\n', )
features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']
data['class']=data['target'].map(lambda ind: raw_data['target_names'][ind])
data.head()
for i in data.target.unique():
    sns.distplot(data['mean area'][data.target==i],
                 kde=1,label='{}'.format(i))

plt.legend()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data_train = scalar.fit_transform(data_train)
data_test = scalar.fit_transform(data_test)

from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(10,activation='softmax'),
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(data_train, label_train, epochs=10, verbose = 2, validation_data = (data_test, label_test))

test_loss, test_acc = model.evaluate(data_test, label_test, verbose=2)
print('\nТочность на проверочных данных: ', test_acc)