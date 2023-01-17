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

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

suid = pd.read_csv("..//input//phpNzIybr.csv")
X = suid.iloc[:30000,1:35].values

Y = suid.iloc[:30000,0].values
G = Y

X





X
Q = [[0 for x in range(len(X[0]))] for y in range(len(X))] 
W = [[0 for x in range(len(X[0]))] for y in range(len(X))]

for x in range(0,len(X)):
    for i in range(0,len(X[0])):
        
        
        Q[x][i] = X[x][i][:3]
        W[x][i] = X[x][i][3:]


        

for x in range(0,len(X)):
    for i in range(0,len(X[0])):
        X[x][i] = float(Q[x][i]) + float(W[x][i])
X    


import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import numpy as np
from IPython.display import Image
import sys
print(os.listdir("../input"))
import keras
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras


p =[100]
for o in p:
    suid = pd.read_csv("..//input//phpNzIybr.csv")
    X = suid.iloc[:40000,1:o].values
    Y = suid.iloc[:40000,0].values
    G = Y
    Q = [[0 for x in range(len(X[0]))] for y in range(len(X))] 
    W = [[0 for x in range(len(X[0]))] for y in range(len(X))]
    
    for x in range(0,len(X)):
        for i in range(0,len(X[0])):
            
            
            Q[x][i] = X[x][i][:3]
            W[x][i] = X[x][i][3:]
    for x in range(0,len(X)):
        for i in range(0,len(X[0])):
            X[x][i] = float(Q[x][i]) + float(W[x][i])
    i = 0
    for z in Y:
        if z=='{0 -1':
            Y[i] = 0
        else:
            Y[i] = 1
        i = i + 1
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    y_train = (y_train > 0.5)
    from sklearn.neighbors import KNeighborsClassifier
    #classifier = KNeighborsClassifier(n_neighbors = 9)
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    #y_test = (y_test > 0.5)
    #y_pred = (y_pred > 0.5)
    
    from sklearn.metrics import confusion_matrix
    acc = []#0 for t in range(0,len(x))]
    prec = []#0 for t in range(0,len(x))]
    rec = []#0 for t in range(0,len(x))]
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    classifier = Sequential()
    
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = len(X[0])))
    classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 55, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)
    y_pred = classifier.predict(X_test)
    y_test = (y_test > 0.5)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    recall = (cm[1][1])/(cm[1][1]+cm[1][0])
    pre = (cm[1][1])/(cm[1][1]+cm[0][1])
    acc.append(accuracy)
    prec.append(pre)
    rec.append(recall)
    #print("Acccuracy for "  , t , "=" , accuracy)
    #print()
    #print("Recall for "  , t , "=" , recall)
    #print()
    #print("Precision for "  , t , "=" , pre)
    #print()
    print(x)
    print()
    print("Accuracy =",acc)
    print()
    print("Recall =",recall)

i = 0
for z in Y:
    if z=='{0 -1':
        Y[i] = 0
    else:
        Y[i] = 1
    i = i + 1

Y
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
y_train


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = len(X[0])))
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 55, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
y_test = (y_test > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
print(cm)
labels = ['Negative' , 'Positive']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()