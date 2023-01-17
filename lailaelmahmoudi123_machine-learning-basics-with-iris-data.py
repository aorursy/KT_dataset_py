import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
iris = pd.read_csv("../input/iris/Iris.csv")
iris.head(10)
iris.isnull().any().sum()
iris.drop("Id", axis=1, inplace=True)
plt.figure(figsize=(7,4))

sns.heatmap(iris.corr(),annot=True,cmap='summer')
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression



from sklearn import svm



from sklearn.neighbors import KNeighborsClassifier



from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score



from sklearn import metrics
train,test= train_test_split(iris, test_size=0.2)
X_train= train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

y_train= train.Species
X_test= test[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

y_test= test.Species
logreg= LogisticRegression()



logreg.fit(X_train, y_train)



prediction= logreg.predict(X_test)



print("Accuracy of the logistic Regression is", metrics.accuracy_score(prediction, y_test))
model= svm.SVC()



model.fit(X_train, y_train)



prediction= model.predict(X_test)



print("Accuracy of the svc is", metrics.accuracy_score(prediction, y_test))
model=KNeighborsClassifier(n_neighbors=6)



model.fit(X_train,y_train)



prediction=model.predict(X_test)



print('Accuracy of KNeighbors is:',metrics.accuracy_score(prediction,y_test))
model=DecisionTreeClassifier()



model.fit(X_train,y_train)



prediction=model.predict(X_test)



print('Accuracy of decision tree is:',metrics.accuracy_score(prediction,y_test))