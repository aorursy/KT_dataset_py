import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans
data = pd.read_csv("../input/iris/Iris.csv")

data
data.drop("Id", axis=1, inplace=True)
data["Species"].unique()
data["Species"] = data["Species"].map({"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":0})

data
plt.scatter(data['SepalLengthCm'],data['SepalWidthCm'])

plt.xlabel('Lenght of sepal')

plt.ylabel('Width of sepal')

plt.show()
x = data.drop("Species", axis=1)

x
kmeans = KMeans(3)

kmeans.fit(x)
clusters = data.drop("Species",axis=1)

clusters["Clusters"] = kmeans.fit_predict(x)
plt.scatter(clusters["SepalLengthCm"], clusters["SepalWidthCm"], c = clusters["Clusters"], cmap="rainbow")

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.show()
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)
kmeans_scaled = KMeans(3)

kmeans_scaled.fit(x_scaled)
clusters_scaled = data.drop("Species",axis=1)

clusters_scaled["Clusters"] = kmeans_scaled.fit_predict(x_scaled)

clusters_scaled
plt.scatter(clusters_scaled["SepalLengthCm"], clusters_scaled["SepalWidthCm"], c=clusters_scaled["Clusters"], cmap="rainbow")

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.show()
data["Predictions"] = clusters_scaled["Clusters"]

data
true = data[data["Predictions"]-data["Species"]==0]["Predictions"].count()

sample = data["Predictions"].count()
accuracy = (true/sample)*100
print ("Accuracy:", "%.2f" % accuracy)