

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



plt.style.use("ggplot")



import os

print(os.listdir("../input"))



iris_data=pd.read_csv("../input/Iris.csv")

iris_data.head()

iris_data.shape
iris_data.info()
iris_data.describe()
fig, ax = plt.subplots()



sns.countplot(x='Species', data=iris_data, palette='RdBu')



plt.show()

sns.pairplot(iris_data, hue="Species",palette="Set1")

plt.show()
y=iris_data.Species

X=iris_data.drop('Species',axis=1).values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
knn=KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(knn.score(X_test, y_test))