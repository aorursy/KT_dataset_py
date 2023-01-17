# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/Iris.csv')
X = data.drop([ 'Id','Species'], axis=1)

Y=data['Species']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y= le.fit_transform(Y)


from sklearn.model_selection  import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, )

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print("r2 value :")

print(r2_score(Y, knn.predict(X)) )
cm = confusion_matrix(y_test,y_pred)

print(cm)



fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix of the knn ')

fig.colorbar(cax)

k_value = list(range(1,50))

accuracy= []

for k in k_value:

    knn = KNeighborsClassifier(n_neighbors=k , metric='minkowski')

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    accuracy.append(accuracy_score(y_test, y_pred))

    

plt.plot(k_value,accuracy )

plt.xlabel('K VALUE ')

plt.ylabel('ACCURACY')

plt.title('GRAPHİC FOR BETTER K VALUE')
