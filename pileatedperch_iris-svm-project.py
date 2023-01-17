import pandas as pd
iris = pd.read_csv('../input/Iris.csv')
iris.head()
iris.rename(columns={'SepalLengthCm':'sepal_length', 'SepalWidthCm':'sepal_width', 'PetalLengthCm':'petal_length', 'PetalWidthCm':'petal_width', 'Species':'species'}, inplace=True)
iris.drop(labels='Id', axis=1, inplace=True)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(iris,hue='species',palette='Dark2')
sns.set_style('whitegrid')
iris_setosa = iris[iris['species']=='Iris-setosa']
sns.kdeplot(iris_setosa['sepal_width'], iris_setosa['sepal_length'],shade=True,cmap='plasma',shade_lowest=False)
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
preds = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
matthews_corrcoef(y_test,preds)
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
pwr = [10.0**(i/3.0) for i in range(-8,9)]
pwr
param_grid = {'C':pwr, 'gamma':pwr}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
grid.best_params_
preds_grid = grid.predict(X_test)
print(confusion_matrix(y_test,preds_grid))
print(classification_report(y_test,preds_grid))
matthews_corrcoef(y_test,preds_grid)