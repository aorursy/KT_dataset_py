# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/train"))
print(os.listdir("../input/tests"))
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
def read_csv(filename):
    train = pd.read_csv(filename)
    x_file = train[["X","Y"]].values
    y_file = train["class"].values

    #print (x_train.size)
    #print (y_train.size)

    colors = {0:'red',1:'blue'}

    plt.scatter(x_file[:,0],x_file[:,1],c=train["class"].apply(lambda x: colors[x]))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return x_file, y_file
#Methode zum ermitteln des Scores bei bestimmten Parametern
def svm(kernel, C, gamma, degree, X_Train, Y_Train, X_Test, Y_Test):
    from sklearn import svm
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model.fit(X_Train,Y_Train)
    return model.score(X_Test,Y_Test)
print("Zeige Trainingsdatensatz")
x_train, y_train = read_csv('../input/train/train.csv')
print("Zeige Testdatensatz")
x_test, y_test = test = read_csv('../input/tests/test.csv')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print (x_train.size)
print (x_test.size)
#C=7 kernel=poly gamma=1 degree=50
kernel='poly'
C=7
gamma=1
degree=50
score=svm(kernel, C, gamma, degree, x_train, y_train, x_test, y_test)
score = score*100
print (str(score.round(2)) + "%")