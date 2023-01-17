import pandas as pd                 # pandas is a dataframe library

import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

import numpy as np

import theano 

import keras 

import tensorflow

%matplotlib inline

from sklearn import metrics

import matplotlib.pyplot as plt # side-stepping mpl backend

import matplotlib.gridspec as gridspec # subplots

import seaborn as sns # Danker visuals

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers.core import Dropout, Flatten, Activation, Dense

from keras.layers.convolutional import Convolution2D, Convolution1D,MaxPooling1D

#Import models from scikit learn module:

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.read_csv("../input/data.csv",header=0)

df.shape
df = df.sample(n=30000 , replace=True)

df.shape
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)

# size of the dataframe

len(df)
df.diagnosis.unique()

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df.head(5)
num_obs = len(df)

print (num_obs)

num_true = len(df.loc[df['diagnosis'] == 1])

num_false = len(df.loc[df['diagnosis'] == 0])

print("Number of Malignant cases:  {0} ({1:2.2f}%)".format(num_true, (float (num_true)/num_obs) * 100))

print("Number of Benign cases: {0} ({1:2.2f}%)".format(num_false, (float(num_false)/num_obs) * 100))
from sklearn.cross_validation import train_test_split

feature_col_names = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

predicted_class_names = ['diagnosis']

X = df[feature_col_names].values     

y = df[predicted_class_names].values 

split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 

# test_size = 0.3 is 30%, 42 is the answer to everything
model = Sequential()

model.add(Dense(512, input_dim=30, init='uniform', activation='sigmoid'))

#model.add(Dense(30, init='uniform', activation='sigmoid'))

model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=600, batch_size=512,  verbose=2 ,validation_data=(X_test, y_test))