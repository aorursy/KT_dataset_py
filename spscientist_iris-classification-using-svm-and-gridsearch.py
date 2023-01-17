import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
data = pd.read_csv("../input/Iris.csv")
data.head()
data.info()
sns.pairplot(data, hue = 'Species');
versicolor = data[data['Species'] == 'Iris-versicolor']
sns.kdeplot( versicolor['SepalWidthCm'], versicolor['SepalLengthCm'], 
            cmap='plasma', shade=True, shade_lowest=False)
from sklearn.model_selection import train_test_split
X = data.drop('Species', axis = 1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
svm.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
grid.fit(X_train, y_train)
pred_grid = grid.predict(X_test)
print(confusion_matrix(y_test, pred_grid))
print(classification_report(y_test, pred_grid))