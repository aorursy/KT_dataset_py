import seaborn as sns
iris = sns.load_dataset('iris')
import pandas as pd
import numpy as np
%matplotlib inline
iris.head()
sns.pairplot(iris,hue='species',palette='Dark2')
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 101)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))