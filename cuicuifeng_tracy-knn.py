import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

%pylab inline

%matplotlib inline

pylab.rcParams['figure.figsize'] = (10,6)
## Import Data

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()
iris.sample(5)
iris.info()
sns.pairplot(iris, hue='PetalWidthCm',palette='coolwarm')
X = iris.iloc[:, 1:5]

X.head()
iris.Species.unique()
from sklearn.preprocessing import LabelEncoder

y = iris.iloc[:, -1]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print(y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

classifier = KNeighborsClassifier(n_neighbors = 10)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print('Accuracy Score:')

print(metrics.accuracy_score(y_test, predictions))