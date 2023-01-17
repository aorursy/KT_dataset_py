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
X=np.loadtxt('../input/Inputs-920.txt')
y=np.loadtxt('../input/Outputs-920.txt')
print(X)
X[:,0]
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.20, random_state=42)
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(100, input_dim=6, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(26))
model.summary()
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train, y_train, epochs=200)
t_pred= model.predict(X_test)
score_test = np.sqrt(mean_squared_error(y_test,t_pred))
print (score_test)
tr_pred= model.predict(X_train)
score_train = np.sqrt(mean_squared_error(y_train,tr_pred))
print (score_train)
print(r2_score(y_train,tr_pred,multioutput='variance_weighted'))
print(r2_score(y_test,t_pred,multioutput='variance_weighted'))