#It's my first attempt at deep learning so it's a little simple... Simple data clean, simple preprocessing and a simple model with a score of 0.76...

#I will keep modifying the model to reach a higher accuracy
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
from tensorflow import keras

from tensorflow.keras import layers

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

from matplotlib import rcParams

%matplotlib inline

import seaborn as sns

import re



#setting a universal figure size

rcParams['figure.figsize'] = 10, 8
#load the train data

data_raw = pd.read_csv('../input/titanic/train.csv')



#load the test data

data_val = pd.read_csv('../input/titanic/test.csv')



#copy the data

data1 = data_raw.copy(deep = True)



#prepare the data

data_cleaner = [data1, data_val]



print(data_raw.info())

data_raw.sample(10)
data_raw.describe(include = 'all')
#complete the missing value in train and test

for dataset in data_cleaner:

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    

    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    

    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)



#delete the unnecessary information

drop_column = ['PassengerId', 'Cabin', 'Ticket', 'Name']

data1.drop(drop_column, axis = 1, inplace = True)
#deal with the continously value



#age

interval = [0, 5, 12, 18, 25, 35, 60, 120]



cats = ['babies', 'children', 'teen', 'student', 'young', 'adult', 'senior']



#fare

quant = [-1, 0, 8, 15, 31, 600]



label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']



for dataset in data_cleaner:

    

    #bins equally

    dataset['FareBin'] = pd.cut(dataset['Fare'], quant, labels = label_quants)

    

    #bins age

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), interval, labels = cats)



    

#delete Age and Fare

del data1['Fare']

del data1['Age']



#preview data again

data1.info()

data_val.info()

data1.sample(10)
#same for the data_val

del data_val['Fare']

del data_val['Name']

del data_val['Ticket']

del data_val['Cabin']

del data_val['Age']



data_val.sample(10)



#finish the data clean
#Preprocessing:

data1.head()
data1 = pd.get_dummies(data1, columns = ['Sex', 'Embarked', 'AgeBin', 'FareBin'])

data_val = pd.get_dummies(data_val, columns = ['Sex', 'Embarked', 'AgeBin', 'FareBin'])

data_val.sample(10)
data1.sample(10)
data1.shape
#re-check the data

print('Train columns with null values: \n', data1.isnull().sum())

print("-"*10)

print (data1.info())

print("-"*10)



print('Test/Validation columns with null values: \n', data_val.isnull().sum())

print("-"*10)

print (data_val.info())

print("-"*10)



data_raw.describe(include = 'all')
#preprare the x_train y_train and x_test

train = data1.drop(['Survived'], axis = 1)

train_ = data1['Survived']



test_ = data_val.drop(['PassengerId'], axis = 1)



X_train = train.values

y_train = train_.values



X_test = test_.values.astype(np.float64, copy = False)



print(X_train)

type(y_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



X_train.shape
X_test.shape
#Modeling

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras

from keras.optimizers import SGD

import graphviz
model = Sequential()



model.add(Dense(18, activation = 'relu', input_dim = 20, kernel_initializer = 'uniform'))



#18:number of hidden units



#dropout layer for previning from overfitting

model.add(Dropout(0.5))



#add second hidden layer

model.add(Dense(60, activation = 'relu', kernel_initializer = 'uniform')) #dont need to pass the input_dim. Keras will solve it



#add another dropout layer

model.add(Dropout(0.5))



#add the output layer

model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) #to obtain the probability



model.summary()
#define a loss function

sgd = SGD(lr = 0.01, momentum = 0.9)



#compile the model

model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])



#fitting the model

model.fit(X_train, y_train, batch_size = 60, epochs = 30, verbose  = 2)
#predicting



y_pred = model.predict(X_test)

NN_pred = (y_pred > 0.5).astype(int)

submission = pd.read_csv("../input/titanic/gender_submission.csv")

submission['Survived'] = NN_pred

submission.to_csv('NN_submission.csv')