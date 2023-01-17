# The Iris Setosa

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'

print(url)

Image(url = url,width=300, height=300)
# The Iris Versicolor

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'

Image(url=url,width=300, height=300)
# The Iris Virginica

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

Image(url=url,width=300, height=300)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



iris = sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris, hue = 'species')
sns.kdeplot(iris[iris['species']=='setosa'][['sepal_length', 'sepal_width']])
from sklearn.model_selection import train_test_split
x = iris.drop('species',axis = 1)

y = iris['species']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
from sklearn.svm import SVC



svm = SVC()

svm.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix, classification_report



prediction = svm.predict(x_test)



print(confusion_matrix(prediction, y_test))

print('\n','-------------------')

print(classification_report(prediction, y_test))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(x_train,y_train)
prediction_best = grid.best_estimator_.predict(x_test)
print(confusion_matrix(prediction_best, y_test))

print('\n','-------------------')

print(classification_report(prediction_best, y_test))