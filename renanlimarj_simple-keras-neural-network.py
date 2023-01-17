# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# Any results you write to the current directory are saved as output.
df0 = pd.read_csv("/kaggle/input/emg-4/0.csv", header=None )
df1 = pd.read_csv("/kaggle/input/emg-4/1.csv", header=None )
df2 = pd.read_csv("/kaggle/input/emg-4/2.csv", header=None )
df3 = pd.read_csv("/kaggle/input/emg-4/3.csv", header=None )
df = pd.concat([df0,df1,df2,df3], axis = 0)
dfx = df.drop(64, axis = 1)
y = df[64]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dfx,
                                                   y,
                                                   test_size=0.25,
                                                   random_state=1)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#evaluation function
def evaluate(y_pred,y_test):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    return list(zip(y_pred,y_test))
# normalizando os dados
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train))
x_test = pd.DataFrame(sc.transform(x_test))
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
classificator = Sequential()
classificator.add(Dense(units = 34, activation = 'softmax', input_dim = 64))
classificator.add(Dense(units = 34, activation = 'softmax'))
classificator.add(Dense(units = 4, activation = 'sigmoid'))
classificator.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy',metrics.categorical_accuracy])



classificator.fit(x_train,y_train , batch_size = 10, nb_epoch = 150)

y_pred = classificator.predict(x_test)
keras.metrics.accuracy(y_test, y_pred)