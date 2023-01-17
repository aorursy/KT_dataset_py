# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import Iris Data 

iris = pd.read_csv('../input/Iris.csv')
iris.head(5)
iris["Species"].value_counts()
#Plot Scatter   PetalLengthCm and PetalWidthCm

iris.plot(kind = "scatter", x = "PetalLengthCm" , y = "PetalWidthCm" )



sns.jointplot(x="PetalLengthCm", y="PetalWidthCm", data=iris, size=5)



sns.FacetGrid(iris, hue="Species", size=5) .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") .add_legend()
#Plot Scatter   SepalLengthCm and SepalWidthCm

iris.plot(kind = "scatter", x = "SepalLengthCm" , y = "SepalWidthCm" )



sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)



sns.FacetGrid(iris, hue="Species", size=5) .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") .add_legend()
#Create Field Target a Number from Species  

def get_target_iris(Species) : 

    if Species == "Iris-setosa" :

        return int(0)

    if Species == "Iris-versicolor" :

        return int(1)

    if Species == "Iris-virginica" :

        return int(2)
iris['Target_Species'] = iris.Species.map(get_target_iris)
iris.head(5)
# You can use "map"  to create field Tarage_Species

iris["Target_Species_2"] = iris["Species"].map({"Iris-setosa" : 0 , "Iris-versicolor" : 1, "Iris-virginica" : 2} )
iris.head(5)
#Prepare Data Test

iris_data = iris.loc[ 1: , ['SepalLengthCm', 'SepalWidthCm'  ,'PetalLengthCm','PetalWidthCm']]

iris_data = np.array(iris_data)

iris_data[:5]
#Prepare Data   Taget 

iris_target = iris.loc[ 1: , ['Target_Species']]

iris_target = np.array(iris_target)

iris_target = np.ravel(iris_target)

iris_target
X = iris_data

y = iris_target
#Step 1 : Import Ski -learn

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
#Step 2 : Instantiate an estimator (instantiate)  Sample K = 5

knn = KNeighborsClassifier(n_neighbors=5)
#Step 3 : Fit The model

knn.fit(X, y)
#Step 4 : Make a prediction (predict)

X_new = [[3, 5, 4, 2]]

knn.predict(X_new)