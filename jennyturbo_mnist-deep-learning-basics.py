# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing data
data = pd.read_csv('../input/train.csv')
print(data.info())
print(data.describe())
print(data.head(3))

test_data = pd.read_csv('../input/test.csv')
print(test_data.info())
print(test_data.describe())
print(data.head(3))
y = data[['label']]
print(y.head())
X=data.drop(['label'], axis = 1)
print(X.head())

#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scalling the X_train set before applying it to NN

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
print(len(y_test))
print(test_data)

test_data_sc = sc.transform(test_data)

print(test_data_sc)
from keras.utils import np_utils

##encode class values as integers (already an integer in this case)
#encoder = LabelEncoder()
#encoder.fit(y_train)
#encoded_y = encoder.transform(y_train)

dummy_ytrain = np_utils.to_categorical(y_train)
print(dummy_ytrain)
from keras.callbacks import EarlyStopping
#early stopping to avoid overfitting
early_stopping = EarlyStopping(patience = 5)
#define baseline model

def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(350, input_dim = 784, kernel_initializer ="normal", activation ="relu"))
    model.add(Dense(350, kernel_initializer ="normal", activation ="relu"))
    model.add(Dense(10, kernel_initializer ="normal", activation ="softmax"))
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
#Debugging is turned off when training by setting verbose to 0.
estimator = KerasClassifier(build_fn = baseline_model, epochs = 10, batch_size = 30, verbose = 1)
#set k fold 
kfold = KFold( n = len(X_train), n_folds = 3, shuffle = True, random_state = 0)

#cross validating the scors

results = cross_val_score(estimator, X_train, dummy_ytrain, cv = kfold, fit_params={'callbacks': [EarlyStopping(patience = 5)]})
print("Accuracy: %0.2f%% (%0.2f%%)" % (results.mean()*100, results.std()* 100))
estimator.fit(X_train, dummy_ytrain, epochs = 10, batch_size = 30)
y_pred = estimator.predict(test_data_sc)
print(y_pred)

print(y_pred.shape)

d = {'Label':y_pred}
df = pd.DataFrame(d)
print(df.head())

df.index.name = 'ImageId'
print(df.head())
df.to_csv('MNIST_output.csv')