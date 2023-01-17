# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
dataset.head(5)
data_features = dataset.iloc[:, 1:8].values
data_target = dataset.iloc[:, 8].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(data_features, data_target, test_size = 0.3,
                                                 random_state = 101)
standard_scaler = StandardScaler()
standard_scaler.fit_transform(X_train)
standard_scaler.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform', input_dim = 7))
classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(units = 1,  activation = 'sigmoid', kernel_initializer = 'uniform'))

classifier.compile(optimizer= 'adam', loss = 'mse',metrics = ['mean_squared_error'])
classifier.fit(X_train, y_train, epochs = 100, batch_size = 10)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import mean_squared_error
cm = mean_squared_error(y_test, y_pred)
cm
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform', input_dim = 7))
    classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 4,  activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1,  activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer= 'adam', loss = 'mse',metrics = ['mean_squared_error'])
    return classifier
new_classifier = KerasRegressor(build_fn= build_classifier, epochs =100, batch_size =10)
score = cross_val_score(estimator = new_classifier, X= X_train, y= y_train, cv= 10, n_jobs= -1)
score
