#import important libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

%matplotlib inline
#load the data

d = datasets.load_iris()

data = pd.DataFrame(d.data, columns=d.feature_names)

data['class'] = d.target

d.target_names
data.shape
data.describe()
sns.boxplot(data=data, width=0.5, fliersize=6)

sns.set(rc={'figure.figsize':(5,5)})
cor = data.corr()

sns.heatmap(cor)

plt.show()
x = data.values[:,:4]

y = data.values[:,4]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=40)

model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))
data
model.predict([[5,4,4.0,2.0]])