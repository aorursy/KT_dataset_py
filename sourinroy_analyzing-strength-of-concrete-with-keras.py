# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv('/kaggle/input/concrete_data.csv')
df.head()
df.shape
df.isnull().sum().sum()
features = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
            'Superplasticizer','Coarse Aggregate', 'Fine Aggregate', 'Age']
X = df[features]
y = df['Strength']
n_features = len(features)
def baseline_model(n_features):
    
    model = Sequential()
    model.add(Dense(10, activation = 'relu', input_shape = (n_features,)))
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
mse_list = []

for i in range(0, 50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    my_model = baseline_model(n_features)
    my_model.fit(X_train, y_train, epochs = 50, verbose = 1)
    preds = my_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mse_list.append(mse)
#Saving and loading the model
my_model.save('BASELINE_MODEL_extended.h5')
#my_model = keras.models.load_model('BASELINE_MODEL_extended.h5')
from statistics import mean, stdev

print('Mean of MSE - ',mean(mse_list))
print('Standard Deviation of MSE -',stdev(mse_list))
from sklearn import preprocessing
mse_list = []

for i in range(0,50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    #Normalizing the training and testing data with sklearn.StandardScaler
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_test)

    my_model = baseline_model(n_features)
    my_model.fit(X_train, y_train, epochs = 50, verbose = 1)
    preds = my_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mse_list.append(mse)
print('Mean of MSE - ',mean(mse_list))
print('Standard Deviation of MSE -',stdev(mse_list))
mse_list = []

for i in range(0,50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    #Normalizing the training and testing data with sklearn.StandardScaler
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_test)

    my_model = baseline_model(n_features)
    my_model.fit(X_train, y_train, epochs = 100, verbose = 1)
    preds = my_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mse_list.append(mse)
print('Mean of MSE - ',mean(mse_list))
print('Standard Deviation of MSE -',stdev(mse_list))
def modified_baseline_model(n_features):
    
    model = Sequential()
    model.add(Dense(10, activation = 'relu', input_shape = (n_features,)))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model
from sklearn import preprocessing
mse_list = []

for i in range(0,50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    #Normalizing the training and testing data with sklearn.StandardScaler
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_test)

    my_model = modified_baseline_model(n_features)
    my_model.fit(X_train, y_train, epochs = 50, verbose = 1)
    preds = my_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mse_list.append(mse)
print('Mean of MSE - ',mean(mse_list))
print('Standard Deviation of MSE -',stdev(mse_list))