#import required packages

import numpy as np 

import pandas as pd 

import scipy as si

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

import keras

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
# Input data files are available in the "../input/" directory.

# To list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Read train & test csv files

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
#EDA analysis for train & test data

train.head()
test.head()
#Numerical variables descriptive analysis

train.describe()
#Categorical variable analysis

train.describe(include=['O'])
#To get descriptor inforamtion

train.info()
# Data preparation for keras model

# To keep only objects

train_obj = train.select_dtypes(include=['object']).copy()

train_obj.head()
# To check the frequency

train['Sex'].value_counts()
train['Embarked'].value_counts()
train=train.replace(["male","female"],[0,1])
train.head()
train=train.replace(['S','C','Q'],[0,1,2])
train= train.fillna(0)

train.head()
train.info()
x=train[["PassengerId","Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]   

y=train[["Survived"]]

x = x.astype(np.float32).values

y = y.astype(np.float32).values
#Test data preparation

test = test.replace(["male", "female"], [0,1])

test = test.replace(["S", "C", "Q"], [0,1,2])

test= test.fillna(0)

x_test=test[["PassengerId","Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]  

x_test.head()
# Import Keras packages

from keras.models import Sequential

from keras.layers import Dense

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.wrappers.scikit_learn import KerasRegressor

# data split

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import StratifiedKFold
seed = 1234

np.random.seed(seed)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33)
x.shape[:]
y.shape[:]
x_train.shape[:]
x_test.shape[:]
y_train.shape[:]
y_test.shape[:]
# Define Model

model = Sequential()

#input layer

model.add(Dense(8, input_dim=(8)))

model.add(Activation("relu"))

# hidden layers1

model.add(Dense(8))

model.add(Activation("relu"))

# hidden layers2

model.add(Dense(8, input_dim=(8)))

model.add(Activation("relu"))

# output layer

model.add(Dense(1, activation='sigmoid'))
#Complie

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Learning

model.fit(x, y, nb_epoch=100, batch_size=10)
predict = np.round(model.predict(x_test))
predictions = pd.DataFrame(predict)
predictions.head()
titanic_sub=pd.concat([test[["PassengerId"]], predictions], axis = 1)

titanic_sub=titanic_sub.rename(columns={0:'Survived'})
titanic_sub.head()
titanic_sub.to_csv("titanic_sub.csv", index=False)