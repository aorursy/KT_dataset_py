#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Read the dataset 
iris=pd.read_csv('../input/iris/Iris.csv')

X=iris.iloc[:,:4]

y=iris.iloc[:,-1]
X = preprocessing.StandardScaler().fit_transform(X)
X[0:4]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
y_test.shape
knnmodel=KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train,y_train)
y_predict1=knnmodel.predict(X_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
acc
from sklearn.metrics import classification_report


print(classification_report(y_test,y_predict1))