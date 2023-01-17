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
data=pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")

data.head()
data.describe()
x=data.iloc[:,0:8]

y=data.iloc[:,8:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(max_depth=4)

rfc.fit(X_train,y_train)

ypred=rfc.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(64, input_dim=8, kernel_initializer='normal', activation='relu'))

model.add(Dense(16,kernel_initializer='normal', activation='relu'))

model.add(Dense(4,kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scale_x=scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(scale_x, y, test_size=0.33, random_state=41)
model.fit(x=X_train,y=y_train,epochs=5)

ypred=model.predict(X_test)

ypred=ypred>=0.51

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))