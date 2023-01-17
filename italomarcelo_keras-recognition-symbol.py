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
# TensorFlow e tf.keras

import tensorflow as tf

from tensorflow import keras



# Librariesauxiliares

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/currency-symbol-datasets/datasets.csv')
def eda(dfA, all=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null: {dfA.isnull().sum().sum()}')

    print(f'{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if all:  # here you put yours prefered analysis that detail more your dataset



        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))



        # print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead:\n{dfA.head()}')

        print(f'\nSamples:\n{dfA.sample(2)}')

        print(f'\nTail:\n{dfA.tail()}')
eda(df)
df.sample(5)
varFeat = df.iloc[:, 1:].columns.tolist()

varT = 'label'
targetNames = df.label.unique().tolist()

targetNames
dataset = df[varFeat]

dataset['target'] = df.label.apply(lambda x: targetNames.index(x))
dataset.sample()
from sklearn.model_selection import train_test_split
X = dataset[varFeat]

y = dataset.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape
train_images = []

for i in range(0,len(Xtrain)):

    x = np.array(Xtrain.iloc[i].values)

    x = x.reshape(64,64)

    train_images.append(x)

train_images = np.asarray(train_images) 
plt.figure(figsize=(10,10))

for i in range(0,50):

    plt.subplot(5,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

#     print(ytrain.iloc[i])

    plt.xlabel(ytrain.iloc[i])

plt.show()
test_images = []

for i in range(0,len(Xtest)):

    x = np.array(Xtest.iloc[i].values)

    x = x.reshape(64,64)

    test_images.append(x)

test_images = np.asarray(test_images) 
plt.figure(figsize=(10,10))

for i in range(0,50):

    plt.subplot(5,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel(ytest.iloc[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(64, 64)),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images, ytrain, epochs=10)
predictions = model.predict(Xtest)
for i in range(0,12):

    print(np.argmax(predictions[i]), targetNames[np.argmax(predictions[i])])
plt.figure(figsize=(10,10))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel(targetNames[np.argmax(predictions[i])])

plt.show()