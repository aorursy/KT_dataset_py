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

        x = filename.split('.')

        if len(x)>1 and x[1] == 'csv':

             print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# TensorFlow e tf.keras

import tensorflow as tf

from tensorflow import keras



# Libraries auxiliares

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2



print(tf.__version__)
dirKaggle = '/kaggle/input/lego-minifigures-classification/'

df = pd.read_csv(f'{dirKaggle}index.csv')
def edaFromData(dfA, allEDA=False, desc='Exploratory Data Analysis'):

    print('Explorando os dados')

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null:\n{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if allEDA:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))

        

        #print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead dos dados:\n{dfA.head()}')

        print(f'\nSamples dos dados:\n{dfA.sample(2)}')

        print(f'\nTail dos dados:\n{dfA.tail()}')
edaFromData(df)
df.sample(3)
dfcopy = df.copy()
df = df.drop(columns=['Unnamed: 0', 'train-valid', 'version'])
df['imageNum'] = df.path.apply(lambda x: cv2.imread(dirKaggle+x, 0))
df.sample(5)
from sklearn.model_selection import train_test_split

X = df['imageNum']

y = df['class_id']

Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=123)
Xtreino.shape, Xteste.shape, ytreino.shape, yteste.shape
ytreino.sample()
plt.figure(figsize=(10,10))

i=0

for img in Xtreino.sample(20):

    plt.subplot(4,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img)

    plt.xlabel('class id')

    i += 1

plt.show()

dfm = pd.read_csv(f'{dirKaggle}metadata.csv')
edaFromData(dfm)
dfm.sample(3)
dfm = dfm.drop(columns='Unnamed: 0')

dfm.sample(3)
for each in Xtreino.head(2).index.values:

    print(each, Xtreino[each], ytreino[each])
plt.figure(figsize=(15,15))

i=0

for each in Xtreino.sample(40).index.values:

    img = Xtreino[each]

    plt.subplot(8,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img)

    label = dfm[dfm.class_id == ytreino[each]]['minifigure_name'].values[0]

    plt.xlabel(label)

    i += 1

plt.show()
plt.figure(figsize=(15,15))

i=0

for each in Xteste.sample(6).index.values:

    img = Xteste[each]

    plt.subplot(2,3,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img)

    label = dfm[dfm.class_id == yteste[each]]['minifigure_name'].values[0]

    plt.xlabel(label)

    i += 1

plt.show()
train_i = []

for tr in Xtreino:

    train_i.append(tr)

train_i = np.asarray(train_i) 

train_i.shape
test_i = []

for tr in Xteste:

    test_i.append(tr)

test_i = np.asarray(test_i) 

test_i.shape
train_i[0], test_i[0]
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(512, 512)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_i, ytreino, epochs=43)
predictions = model.predict(test_i)
np.argmax(predictions[2])
plt.figure(figsize=(10,10))

for i in range(6):

    plt.subplot(3,2,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_i[i], cmap=plt.cm.binary)

    label = dfm[dfm.class_id == np.argmax(predictions[i])]['minifigure_name'].values[0]

    plt.xlabel(label)

plt.show()