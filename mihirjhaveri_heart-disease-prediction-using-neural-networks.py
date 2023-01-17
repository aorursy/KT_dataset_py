# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



import sys

import pandas as pd

import numpy as np

import sklearn

import matplotlib

import keras



print('Python: {}'.format(sys.version))

print('Numpy: {}'.format(np.__version__))

print('Sklearn: {}'.format(sklearn.__version__))

print('Pandas: {}'.format(pd.__version__))

print('Matplotlib: {}'.format(matplotlib.__version__))

print('Keras: {}'.format(keras.__version__))
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
# import the heart disease dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"



# the names will be the names of each column in our pandas DataFrame

names = ['age',

        'sex',

        'cp',

        'trestbps',

        'chol',

        'fbs',

        'restecg',

        'thalach',

        'exang',

        'oldpeak',

        'slope',

        'ca',

        'thal',

        'class']



# read the csv

cleveland = pd.read_csv(url, names=names)
# print the shape of the DataFrame, so we can see how many examples we have

print('Shape of DataFrame: {}'.format(cleveland.shape))

print(cleveland.loc[1])
# print the last twenty or so data points

cleveland.loc[280:]
# remove missing data (indicated with a "?")

data = cleveland[~cleveland.isin(['?'])]

data.loc[280:]
# drop rows with NaN values from DataFrame

data = data.dropna(axis=0)

data.loc[280:]
# print the shape and data type of the dataframe

print(data.shape)

print(data.dtypes)
# transform data to numeric to enable further analysis

data = data.apply(pd.to_numeric)

data.dtypes
# print data characteristics, usings pandas built-in describe() function

data.describe()
# plot histograms for each variable

data.hist(figsize = (12, 12))

plt.show()
# create X and Y datasets for training

from sklearn import model_selection



X = np.array(data.drop(['class'], 1))

y = np.array(data['class'])



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
# convert the data to categorical labels

from keras.utils.np_utils import to_categorical



Y_train = to_categorical(y_train, num_classes=None)

Y_test = to_categorical(y_test, num_classes=None)

print(Y_train.shape)

print(Y_train[:10])
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



# define a function to build the keras model

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, kernel_initializer='normal', activation='relu'))

    model.add(Dense(5, activation='softmax'))

    

    # compile model

    adam = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model



model = create_model()



print(model.summary())
# fit the model to the training data

model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)
# convert into binary classification problem - heart disease or no heart disease

Y_train_binary = y_train.copy()

Y_test_binary = y_test.copy()



Y_train_binary[Y_train_binary > 0] = 1

Y_test_binary[Y_test_binary > 0] = 1



print(Y_train_binary[:20])
# define a new keras model for binary classification

def create_binary_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # Compile model

    adam = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model



binary_model = create_binary_model()



print(binary_model.summary())
# fit the binary model on the training data

binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose = 1)
# generate classification report using predictions for categorical model

from sklearn.metrics import classification_report, accuracy_score



categorical_pred = np.argmax(model.predict(X_test), axis=1)



print('Results for Categorical Model')

print(accuracy_score(y_test, categorical_pred))

print(classification_report(y_test, categorical_pred))
# generate classification report using predictions for binary model 

binary_pred = np.round(binary_model.predict(X_test)).astype(int)



print('Results for Binary Model')

print(accuracy_score(Y_test_binary, binary_pred))

print(classification_report(Y_test_binary, binary_pred))