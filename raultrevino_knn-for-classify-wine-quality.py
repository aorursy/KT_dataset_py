import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing,neighbors

from sklearn.model_selection import cross_validate, train_test_split

data = pd.read_csv("../input/whitewine/winequalitywhite.csv",sep=";")

data.head()
data.tail()
data.describe()
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
print(data.dtypes)
print("Correlaciones en el dataset:")

data.corr()
plt.matshow(data.corr())
x = data.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

data_n = pd.DataFrame(x_scaled, columns=data.columns.values)

data_n['quality'] = data['quality']
data_n.head()
data_vars = data.columns.values.tolist()

Y = ['quality']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data_n[X],data_n[Y], test_size=0.30)  
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,Y_train.values.ravel())
accuracy = clf.score(X_test,Y_test)

print("Accuraccy")

accuracy
data_vars = data.columns.values.tolist()

Y = ['quality']

REMOVE = ['quality','fixed acidity','volatile acidity','citric acid','chlorides','residual sugar','free sulfur dioxide','total sulfur dioxide']

X = [v for v in data_vars if v not in REMOVE]

X_train, X_test, Y_train, Y_test = train_test_split(data_n[X],data_n[Y], test_size=0.30)  
clf = neighbors.KNeighborsClassifier(n_neighbors=5)

clf.fit(X_train,Y_train.values.ravel())

accuracy = clf.score(X_test,Y_test)

print("Accuraccy")

accuracy
X_train