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
data=pd.read_csv("/kaggle/input/ph-recognition/ph-data.csv")

data.head()
x=data.iloc[:,0:3]

y=data.iloc[:,3:]
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion= "entropy")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
dtc.fit(x_train,y_train)

ypred=dtc.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scale_x=scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(scale_x, y, test_size=0.25, random_state=42)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(128, input_dim=3, kernel_initializer='normal', activation='relu'))

model.add(Dense(60,kernel_initializer='normal', activation='relu'))

model.add(Dense(15,kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

encoder = LabelEncoder()

encoder.fit(y)

encoded_Y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)

dummy_y
X_train, X_test, y_train, y_test = train_test_split(scale_x, dummy_y, test_size=0.25, random_state=42)
model.fit(X_train,y_train,epochs=200)
ypred2=model.predict(X_test)

ypred=ypred2.argmax(1)

y_test=y_test.argmax(1)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))
model = Sequential()

model.add(Dense(1024, input_dim=3, kernel_initializer='normal', activation='relu'))

model.add(Dense(500,kernel_initializer='normal', activation='relu'))

model.add(Dense(120,kernel_initializer='normal', activation='relu'))

model.add(Dense(60,kernel_initializer='normal', activation='relu'))

model.add(Dense(15,kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y2_train, y2_test = train_test_split(scale_x, dummy_y, test_size=0.25, random_state=42)

model.fit(X_train,y2_train,epochs=200)
ypred3=model.predict(X_test)

ypred=ypred3.argmax(1)

y_test=y2_test.argmax(1)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))
print(data.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))
best=0

best_acc=0

for i in range(1,15):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    ypred=knn.predict(x_test)

    acc=metrik.accuracy_score(y_true=y_test,y_pred=ypred)

    if best_acc<acc:

        best=i

        best_acc=acc

print("Accuracy: "+str(best_acc))

print("neighbour: "+str(best))
ybinary=y>7

ybinary=ybinary.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, ybinary, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

print(metrik.classification_report(y_true=y_test,y_pred=ypred))