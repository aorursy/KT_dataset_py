# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np; np.random.seed(0)

import seaborn as sns; sns.set()

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Import the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

print(train)

train.sample(10)

corr_matrix=train.sample(1000).corr()

print(corr_matrix['label'].sort_values(ascending=False))
colors = ['blue']



data=train.mean()#.info()#.groupby(['pixel1']).mean()

data=data.values[1:]

data=data.reshape(-1,28)

plt.imshow(data)

plt.show()


train_labels = train['label'][0:36300]

train_dataset = train.drop('label',axis=1)[0:36300]

X = np.array(train_dataset).reshape(train_dataset.shape[0],28,28)

Y = np.array(train_labels).reshape(train_labels.shape[0],1)



#Lets create the validation set

train_Valid_labels = train['label'][36300:]

train_Valid_dataset = train.drop('label',axis=1)[36300:]

X_valid=np.array(train_Valid_dataset).reshape(train_Valid_dataset.shape[0],28,28)

Y_valid = np.array(train_Valid_labels).reshape(train_Valid_labels.shape[0],1)
model = keras.models.Sequential([

    #the number of neurons at the begining depends and needs to be 28,28 

keras.layers.Flatten(input_shape=[28, 28]),

keras.layers.Dense(300, activation="relu"),

keras.layers.Dense(100, activation="relu"),#The parameters are just the one in the book: 

    #hands on machine learning with

    #scikit learn and theano

    

keras.layers.Dense(10, activation="softmax")

    #the number of output in exit needs to be 10 because we have 10 classes

])

model.summary()

model.layers

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history = model.fit(X, Y, epochs=10,validation_data=(X_valid, Y_valid))
#Let's plot the performances

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 3) # set the vertical range to [0-1]

plt.show()
#Let's show instance 1

data=X[0]

data=data.reshape(-1,28)

plt.imshow(data)

plt.show()

print("here is the first instance of our training set")
#Now let's show

X_try = X[0:1]

y_proba = model.predict(X_try)

y_proba.round(2)
model = keras.models.Sequential([

keras.layers.Flatten(input_shape=[28, 28]),

    

keras.layers.Dense(392, activation="relu"),

#keras.layers.MaxPooling2D([28,28]),

#keras.layers.Dropout(0.5),



keras.layers.Dense(10, activation="softmax")

])

model.summary()

model.layers

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history = model.fit(X, Y, epochs=10,validation_data=(X_valid, Y_valid))
#Let's show instance 1



inst=4

for i in [0,1,2,3]:

    data=X[i]

    data=data.reshape(-1,28)

    plt.imshow(data)

    plt.show()

print("here are the first",inst," instances of our training set")

#Now let's show

X_try = X[:4]

y_proba = model.predict(X_try)

print(model.predict_classes(X_try))

y_proba.round(2)