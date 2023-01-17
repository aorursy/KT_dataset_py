import seaborn as sns
import pandas as pd

import numpy as np

%matplotlib inline

iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
iris.head()
sns.pairplot(iris,hue='species')
setosa = iris[iris['species']=='Iris-setosa']

sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)
#Train test split
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)

y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.001,.01]}
grid = GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))