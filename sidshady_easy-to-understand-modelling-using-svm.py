import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



iris = pd.read_csv("../input/Iris.csv")





iris.drop('Id',axis=1,inplace=True)



iris.info()
iris.head()
sns.pairplot(data=iris)
iris.groupby("Species").mean()
sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
sns.boxplot(x='Species',y='SepalWidthCm',data=iris)
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
sns.boxplot(x='Species',y='PetalWidthCm',data=iris)
from sklearn.cross_validation import train_test_split



X_train,X_test,y_train,y_test = train_test_split(iris.drop("Species",axis=1),iris['Species'],test_size=0.3)
from sklearn.svm import SVC 



svc_model = SVC()
svc_model.fit(X_train,y_train)
y_predicts = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print(classification_report(y_test,y_predicts))

print(confusion_matrix(y_test,y_predicts))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
svc_model = SVC(C=10,gamma=0.1)
svc_model.fit(X_train,y_train)
y_predicts = svc_model.predict(X_test)
print(classification_report(y_test,y_predicts))

print(confusion_matrix(y_test,y_predicts))
svc_model.score(X_train,y_train)
svc_model.score(X_test,y_test)