import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

%matplotlib inline

print(check_output(["ls", "../input"]).decode("utf8"))
iris_df = pd.read_csv("../input/Iris.csv")
iris_df.describe()
iris_df.info()
iris_df.head()
print(iris_df.Species.unique())
iris = iris_df.groupby('Species',as_index= False)["Id"].count()

iris
ax = iris_df[iris_df.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris_df[iris_df.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='white', label='versicolor',ax=ax)

iris_df[iris_df.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=ax)

ax.set_xlabel("Sepal Length")

ax.set_ylabel("Sepal Width")

ax.set_title("Relationship between Sepal Length and Width")
petal = np.array(iris_df[["PetalLengthCm","PetalWidthCm"]])

sepal = np.array(iris_df[["SepalLengthCm","SepalWidthCm"]])



key = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

Y = iris_df['Species'].map(key)
from sklearn.cross_validation import train_test_split



X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(sepal,Y,test_size=0.2,random_state=42)



X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(petal,Y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=4,presort=True)

model.fit(X_train_std_S,y_train_S)

print('Training Accuracy Sepal = {}'.format(model.score(X_train_std_S, y_train_S)))

print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std_S, y_test_S)))



model.fit(X_train_std_P,y_train_P)

print('\nTraining Accuracy Petal = {}'.format(model.score(X_train_std_P, y_train_P)))

print('Testing  Accuracy Petal = {}'.format(model.score(X_test_std_P, y_test_P)))