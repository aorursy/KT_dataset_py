import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/kaggle/input/uci-breast-cancer-wisconsin-original/breast-cancer-wisconsin.data.txt")

data.head()
data = data.drop(['1000025'], axis=1)
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    data.index[data[column] == '?'].tolist()

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
print("Numero de registros:"+str(data.shape[0]))

invalid_rows = None

for column in data.columns.values:

    if len(data.index[data[column] == '?'].tolist()) > 0:

        invalid_rows = data.index[data[column] == '?'].tolist()

        data = data.drop(invalid_rows)  

        print(invalid_rows)
data.columns = [ "Clump Thickness ", "Uniformity of Cell Size", "Uniformity of Cell Shape ", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

data.head()
data_vars = data.columns.values.tolist()

Y = ['Class']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)
X_std_train = StandardScaler().fit_transform(X_train[X])

X_std_test =  StandardScaler().fit_transform(X_test[X])
from sklearn.decomposition import PCA 

acp = PCA(.75)

X_reduction_train = acp.fit_transform(X_std_train)

acp = PCA(n_components=len(X_reduction_train[0]))

X_reduction_test = acp.fit_transform(X_std_test)
len(X_reduction_train[0])
clf = neighbors.KNeighborsClassifier()
clf.fit(X_reduction_train,Y_train.values.ravel())
accuracy = clf.score(X_reduction_test,Y_test)

accuracy
data_predict = [ X_reduction_test[0],]

clf.predict(data_predict)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(neighbors.KNeighborsClassifier(),X_reduction_train,Y_train.values.ravel(), scoring="accuracy", cv=20)
scores.mean()