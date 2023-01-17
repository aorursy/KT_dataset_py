import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
iris = pd.read_csv("../input/Iris.csv")
iris.head()
sns.pairplot(iris,hue='Species')
sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris[iris['Species']=='Iris-setosa'][['SepalLengthCm','SepalWidthCm']],kind='kde')
from sklearn.model_selection import train_test_split
X = iris.drop('Species',axis=1)
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(chr(10))
print(classification_report(y_test,predictions))
