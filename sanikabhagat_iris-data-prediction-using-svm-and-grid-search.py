# The Iris Setosa

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'

Image(url,width=300, height=300)
# The Iris Versicolor

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'

Image(url,width=300, height=300)
# The Iris Virginica

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

Image(url,width=300, height=300)
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
iris = pd.read_csv('../input/Iris.csv')
iris.head()
iris.shape
iris.info()
iris.describe()
sns.pairplot(iris.drop("Id",axis=1),hue='Species',palette='Dark2')
setosa = iris[iris['Species']=='Iris-setosa']
sns.kdeplot(setosa['SepalWidthCm'],setosa['SepalLengthCm'],cmap='plasma',shade=True,shade_lowest=False)
X = iris.drop(["Id","Species"],axis=1)

y = iris["Species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
# Instantiating SVM model (basically creating a svm object)
svc_model = SVC()
# Training or fitting the model on training data
svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))