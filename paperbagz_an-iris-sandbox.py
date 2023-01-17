# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from numpy import random

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv("../input/iris/Iris.csv").drop("Id", axis = 1 ,)





#independent feature

indFeat = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
dataset.head()
dataset.info()
dataset.describe()
dataset["Species"].value_counts()
sns.pairplot(dataset, hue = "Species")

plt.show()
plt.figure(figsize = (15,10))

sns.scatterplot(dataset["SepalLengthCm"], dataset["SepalWidthCm"], hue = dataset["Species"], s = 100)

plt.show()
plt.figure(figsize = (15,10))

sns.scatterplot(dataset["PetalLengthCm"], dataset["PetalWidthCm"], hue = dataset["Species"], s = 100)

plt.show()
#Dimensionality reduction



from sklearn.decomposition import PCA



pca = PCA(n_components = 1)

dataDecompose = pd.DataFrame(pca.fit_transform(dataset[["PetalLengthCm", "PetalWidthCm"]]))

dataDecompose["Species"] = dataset["Species"]
plt.figure(figsize = (20,10))

sns.pairplot(dataDecompose, hue = "Species", height = 5)
def NaiveClassifier(data, input = 0):

    

    if input <= data[data["Species"] == "Iris-setosa"].max()[0] and input >= data[data["Species"] == "Iris-setosa"].min()[0]:

        return "Iris-setosa"

    

    elif input < data[data["Species"] == "Iris-virginica"].min()[0]:

        return "Iris-versicolor"

    

    elif input > data[data["Species"] == "Iris-versicolor"].max()[0]:

        return "Iris-virginica"

    

    else:

        return random.choice(["Iris-versicolor", "Iris-virginica"])

    
sumasi = 0

for x in range(5):

    test = []

    for data in dataDecompose[0]:

        test.append(NaiveClassifier(dataDecompose, data))

    

    sumasi +=((pd.Series(test) == dataDecompose["Species"]).astype(int)).sum()
akurasi = sumasi / (150*5)



print("Accuration on Simple Tresholding and Random :", akurasi)
# data partitioning for k=5 fold



def partitioning(dataset, k = 5):

    grouped = list(dataset.groupby("Species"))

    test = []

    train = []

    for i in range (k):

        temp_concat = ""

        temp_concat = pd.concat([grouped[0][1][i*10:(i+1)*10],grouped[1][1][i*10:(i+1)*10],grouped[2][1][i*10:(i+1)*10]])

        test.append(temp_concat)

        train.append(pd.DataFrame(dataset[~(dataset.isin(test[i]))].dropna(how = "all")))

    return test, train



test,train = partitioning(dataset)
from sklearn.naive_bayes import GaussianNB



gaussNB = GaussianNB()



acc = 0

for i in range(5):

    NB = gaussNB.fit( X = train[i][indFeat], y = train[i]["Species"])

    pred = NB.predict(test[i][indFeat])

    acc += (pred == test[i]["Species"]).sum()



acc = acc/150



print("Naive Bayes Accuracy =",acc)
from sklearn.neighbors import KNeighborsClassifier



kNN = KNeighborsClassifier(n_neighbors = 7, metric = "euclidean")



for i in range(5):

    neigh = kNN.fit(X = train[i][indFeat], y = train[i]["Species"])

    pred = neigh.predict(test[i][indFeat])

    acc += (pred == test[i]["Species"]).sum()



acc = acc/150

print("kNN Accuracy=",acc)
from sklearn import svm

VecMachine = svm.SVC(gamma = "auto")

for i in range(5):

    neigh = VecMachine.fit(X = train[i][indFeat], y = train[i]["Species"])

    pred = VecMachine.predict(test[i][indFeat])

    acc += (pred == test[i]["Species"]).sum()



acc = acc/150

print("SVM Accuracy=",acc)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz





tree_model = DecisionTreeClassifier()

tree_model.fit(X = dataset[indFeat], y=dataset['Species'])



print("Decision Tree Accuracy =",(tree_model.predict(dataset[indFeat])

       ==dataset["Species"]).sum()/len(dataset))



dec = export_graphviz(tree_model)



from IPython.display import display



display(graphviz.Source(dec))