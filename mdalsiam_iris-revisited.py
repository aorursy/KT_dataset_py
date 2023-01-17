import numpy as np

import pandas as pd



from sklearn import datasets

iris = datasets.load_iris()



print(iris.keys())
print("Data of iris dataset\n________________________________________________________")

print(iris.data)
print("Targets of iris dataset\n________________________________________________________")

print(iris.target)
print(iris.feature_names)
print(iris.target_names)
print(iris.filename)
print(iris.DESCR)
iris.data.shape
for k in range (150):

  print(k, " ", iris.data[k], " -> ", iris.target[k])
import seaborn as sns

iris = sns.load_dataset('iris')

sns.set(style = "whitegrid", color_codes = True)

sns.pairplot(iris, kind = "reg", hue = "species")
iris.columns
iris.sepal_length
iris.species
correlation = iris.corr()

correlation
sns.heatmap(iris.corr(), cmap = 'coolwarm',  annot=True)
sns.pairplot(iris, size=5, vars=["sepal_length", "petal_length"], \

             markers=["o", "s", "D"], hue="species")
sns.lmplot(x = "sepal_length", y = "petal_length", hue = "species", data = iris, \

              palette = "Set1", markers=["o", "s", "D"])
sns.boxplot(data = iris, orient = "h")
iris["ID"] = iris.index

iris["ratio"] = iris["sepal_length"]/iris["sepal_width"]

sns.lmplot(x="ID", y="ratio", data=iris, \

           hue="species", markers=["o", "x", "D"], fit_reg=False)
sns.stripplot(x = "species", y = "sepal_length", data = iris)
sns.swarmplot(x = "species", y = "sepal_length", data = iris)
sns.lmplot(x = 'sepal_length', y = 'petal_length', hue = 'species', data = iris, fit_reg = False)
sns.lmplot(x = 'sepal_length', y = 'sepal_width', hue = 'species', data = iris, fit_reg = False)
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x = iris.sepal_length

y = iris.sepal_width

z = iris.petal_length



x_title = "Sepal Length"

y_title = "Sepal Width"

z_title = "Petal_Length"



ax.set_xlabel(x_title)

ax.set_ylabel(y_title)

ax.set_zlabel(z_title)



xx = []

yy = []

zz = []



for i in range (0, 50):

  xx.append(x[i])

  yy.append(y[i])

  zz.append(z[i])



ax.scatter(xx, yy, zz, c = 'r', marker = 'o', s = 10)



xx = []

yy = []

zz = []



for i in range (50, 100):

  xx.append(x[i])

  yy.append(y[i])

  zz.append(z[i])



ax.scatter(xx, yy, zz, c = 'g', marker = 'x', s = 10)



xx = []

yy = []

zz = []



for i in range (100, 150):

  xx.append(x[i])

  yy.append(y[i])

  zz.append(z[i])



ax.scatter(xx, yy, zz, c = 'b', marker = 'D', s = 10)



plt.show()
import plotly.express as px

df = px.data.iris()

fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', \

              color='species')

fig.show()
fig = px.scatter(df, x = "sepal_length", y = "sepal_width", color = "species" )

fig.show()
from numpy import percentile

data = iris.sepal_length

quartiles = percentile(data, [25, 50, 75])

data_min, data_max = data.min(), data.max()

print('Min: %.3f' % data_min)

print('Q1: %.3f' % quartiles[0])

print('Median: %.3f' % quartiles[1])

print('Q3: %.3f' % quartiles[2])

print('Max: %.3f' % data_max)
tot = 0

for i in range (150):

  tot += data[i]



avg = tot / data.size

avg 
dev_tot = 0

for i in range (150):

  temp = abs(avg-data[i])

  dev_tot += temp*temp



sigma_sq = dev_tot / iris.sepal_length.size

from math import sqrt

sigma = sqrt(sigma_sq)

sigma
iris = datasets.load_iris()

X = iris.data

y = iris.target
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test);

from sklearn import metrics

print(metrics.accuracy_score(y_test, y_predict))