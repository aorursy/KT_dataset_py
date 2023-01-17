import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
iris = pd.read_csv('../input/iris/Iris.csv')
iris.head(5)
iris.info()
iris.describe()
# Data visualisation
sns.pairplot(iris,hue='Species')
# The species 'Iris-setosa' shown in blue is more distinguishable in its features compared to the other two species of iris.
# Logistic regression to predict the species based on the features



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],iris['Species'],test_size=0.33)
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
# Good result given the small dataset!