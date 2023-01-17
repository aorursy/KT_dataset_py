# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow.keras
df=pd.read_csv("/kaggle/input/concrete-data/concrete_data.csv")
df.head()
X1=df.iloc[:,:-1]
Y1=df.iloc[:,-1]
df.isnull().sum()
from keras.models import Sequential
from keras.layers import Dense
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X1_train.shape[1],)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model=regression_model()
# fit the model
results=[]
for i in range(50):
    X1_train,X1_test,y1_train,y1_test=train_test_split(X1,Y1,test_size=0.3,random_state=42)
    model.fit(X1_train, y1_train, validation_split=0.3, epochs=50, verbose=0)
    y_pred=model.predict(X1_test)
    results.append(mean_squared_error(y1_test, y_pred))

import statistics
print("Results mean"+str(statistics.mean(results)))
print("Results std"+str(statistics.stdev(results)))
X1_norm=(X1-X1.mean())/X1.std()
X1_norm.head()
X2_train,X2_test,y2_train,y2_test=train_test_split(X1_norm,Y1,test_size=0.3,random_state=21)
# fit the model
results_b=[]
for i in range(50):
    X2_train,X2_test,y2_train,y2_test=train_test_split(X1_norm,Y1,test_size=0.3,random_state=42)
    model.fit(X2_train, y2_train, validation_split=0.3, epochs=50, verbose=0)
    y_pred=model.predict(X2_test)
    results_b.append(mean_squared_error(y2_test, y_pred))

import statistics
print("Results mean"+str(statistics.mean(results_b)))
print("Results std"+str(statistics.stdev(results_b)))
# fit the model
results_c=[]
for i in range(50):
    X2_train,X2_test,y2_train,y2_test=train_test_split(X1_norm,Y1,test_size=0.3,random_state=42)
    model.fit(X2_train, y2_train, validation_split=0.3, epochs=100, verbose=0)
    y_pred=model.predict(X2_test)
    results_c.append(mean_squared_error(y2_test, y_pred))

import statistics
print("Results mean"+str(statistics.mean(results_c)))
print("Results std"+str(statistics.stdev(results_c)))
# define regression model
def regression_model_d():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X1_train.shape[1],)))
    model.add(Dense(10, activation='relu', input_shape=(X1_train.shape[1],)))
    model.add(Dense(10, activation='relu', input_shape=(X1_train.shape[1],)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model_d=regression_model_d()
# fit the model
results_d=[]
for i in range(50):
    X2_train,X2_test,y2_train,y2_test=train_test_split(X1_norm,Y1,test_size=0.3,random_state=42)
    model.fit(X2_train, y2_train, validation_split=0.3, epochs=50, verbose=0)
    y_pred=model.predict(X2_test)
    results_d.append(mean_squared_error(y2_test, y_pred))

import statistics
print("Results mean"+str(statistics.mean(results_d)))
print("Results std"+str(statistics.stdev(results_d)))
