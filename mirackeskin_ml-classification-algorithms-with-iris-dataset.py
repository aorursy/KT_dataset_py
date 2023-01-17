# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/iris/Iris.csv")
data.head()
data.info()
data=data.iloc[:,1:6]
data.head()
scatterplot = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=scatterplot)

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=scatterplot)

scatterplot.set_xlabel("Sepal Length")

scatterplot.set_ylabel("Sepal Width")

scatterplot.set_title("Sepal Length VS Petal Width")

scatterplot=plt.gcf()

scatterplot.set_size_inches(10,6)

plt.show()
scatterplot = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=scatterplot)

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=scatterplot)

scatterplot.set_xlabel("Petal Length")

scatterplot.set_ylabel("Petal Width")

scatterplot.set_title("Petal Length VS Petal Width")

scatterplot=plt.gcf()

scatterplot.set_size_inches(10,6)

plt.show()
data.plot()
petallength=data.iloc[:,2:3].values

petalwidth=data.iloc[:,3:4].values



sepallength=data.iloc[:,0:1].values

sepalwidth=data.iloc[:,1:2].values



classes=data.iloc[:,4].values
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(petallength,classes,test_size=0.3,random_state=0)

X_train,X_test,Y_train,Y_test=train_test_split(petalwidth,classes,test_size=0.3,random_state=0)



x_train2,x_test2,y_train2,y_test2=train_test_split(sepallength,classes,test_size=0.3,random_state=0)

X_train2,X_test2,Y_train2,Y_test2=train_test_split(sepalwidth,classes,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression



logr=LogisticRegression()



logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)



logr.fit(X_train,Y_train)

Y_pred=logr.predict(X_test)



logr.fit(x_train2,y_train2)

y_pred2=logr.predict(x_test2)



logr.fit(X_train2,Y_train2)

Y_pred2=logr.predict(X_test2)



print("The Accuracy values for Petal VS Sepal..\n")



print("The Accuracy of the LogisticRegression for PetalLength Feature:",metrics.accuracy_score(y_pred,y_test))

print("The Accuracy of the LogisticRegression for PetalWidth Feature:",metrics.accuracy_score(Y_pred,Y_test))



print("The Accuracy of the LogisticRegression for SepalLength Feature:",metrics.accuracy_score(y_pred2,y_test2))

print("The Accuracy of the LogisticRegression for SepalWidth Feature:",metrics.accuracy_score(Y_pred2,Y_test2))



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')



knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)



knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)



knn.fit(x_train2,y_train2)

y_pred2=knn.predict(x_test2)



knn.fit(X_train2,Y_train2)

Y_pred2=knn.predict(X_test2)



print("The Accuracy values for Petal VS Sepal..\n")



print("The Accuracy of the KNN for PetalLength Feature:",metrics.accuracy_score(y_pred,y_test))

print("The Accuracy of the KNN for PetalWidth Feature:",metrics.accuracy_score(Y_pred,Y_test))



print("The Accuracy of the KNN for SepalLength Feature:",metrics.accuracy_score(y_pred2,y_test2))

print("The Accuracy of the KNN for SepalWidth Feature:",metrics.accuracy_score(Y_pred2,Y_test2))



from sklearn.svm import SVC

svc=SVC(kernel='linear')



svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)



svc.fit(X_train,Y_train)

Y_pred=svc.predict(X_test)



svc.fit(x_train2,y_train2)

y_pred2=svc.predict(x_test2)



svc.fit(X_train2,Y_train2)

Y_pred2=svc.predict(X_test2)



print("The Accuracy values for Petal VS Sepal..\n")



print("The Accuracy of the SVM for PetalLength Feature:",metrics.accuracy_score(y_pred,y_test))

print("The Accuracy of the SVM for PetalWidth Feature:",metrics.accuracy_score(Y_pred,Y_test))



print("The Accuracy of the SVM for SepalLength Feature:",metrics.accuracy_score(y_pred2,y_test2))

print("The Accuracy of the SVM for SepalWidth Feature:",metrics.accuracy_score(Y_pred2,Y_test2))



from sklearn.tree import DecisionTreeClassifier



dtc=DecisionTreeClassifier(criterion='entropy')



dtc.fit(x_train,y_train)

y_pred=dtc.predict(x_test)



dtc.fit(X_train,Y_train)

Y_pred=dtc.predict(X_test)



dtc.fit(x_train2,y_train2)

y_pred2=dtc.predict(x_test2)



dtc.fit(X_train2,Y_train2)

Y_pred2=dtc.predict(X_test2)



print("The Accuracy values for Petal VS Sepal..\n")



print("The Accuracy of the DecisionTree for PetalLength Feature:",metrics.accuracy_score(y_pred,y_test))

print("The Accuracy of the DecisionTree for PetalWidth Feature:",metrics.accuracy_score(Y_pred,Y_test))



print("The Accuracy of the DecisionTree for SepalLength Feature:",metrics.accuracy_score(y_pred2,y_test2))

print("The Accuracy of the DecisionTree for SepalWidth Feature:",metrics.accuracy_score(Y_pred2,Y_test2))
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')



rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)



rfc.fit(X_train,Y_train)

Y_pred=rfc.predict(X_test)



rfc.fit(x_train2,y_train2)

y_pred2=rfc.predict(x_test2)



rfc.fit(X_train2,Y_train2)

Y_pred2=rfc.predict(X_test2)



print("The Accuracy values for Petal VS Sepal..\n")



print("The Accuracy of the RandomForest for PetalLength Feature:",metrics.accuracy_score(y_pred,y_test))

print("The Accuracy of the RandomForest for PetalWidth Feature:",metrics.accuracy_score(Y_pred,Y_test))





print("The Accuracy of the RandomForest for SepalLength Feature:",metrics.accuracy_score(y_pred2,y_test2))

print("The Accuracy of the RandomForest for SepalWidth Feature:",metrics.accuracy_score(Y_pred2,Y_test2))
