# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris_dataframe = pd.read_csv("/kaggle/input/iris/Iris.csv")

iris_dataframe.head()
print(iris_dataframe.columns)

print(iris_dataframe.Species.unique())
print(iris_dataframe.info())
print(iris_dataframe.describe())
Iris_setosa = iris_dataframe[iris_dataframe.Species == "Iris-setosa"]

Iris_versicolor = iris_dataframe[iris_dataframe.Species == "Iris-versicolor"]

Iris_virginica = iris_dataframe[iris_dataframe.Species == "Iris-virginica"]



print(Iris_setosa.describe())
print(Iris_versicolor.describe())
print(Iris_virginica.describe())
iris_dataframe_notId = iris_dataframe.drop(["Id"], axis = 1)
plt.figure(1) # for plotting graphs in different pages.

plt.plot(Iris_setosa["Id"], Iris_setosa["PetalLengthCm"], color = "red", label ="Setosa")

plt.plot(Iris_versicolor["Id"], Iris_versicolor["PetalLengthCm"], color = "green", label ="Versicolor")

plt.plot(Iris_virginica["Id"], Iris_virginica["PetalLengthCm"], color = "blue", label ="Virginica")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend() # put label to the nice coordination on the graph.

plt.show()
plt.figure(2)

plt.scatter(Iris_setosa["Id"], Iris_setosa["PetalLengthCm"], color = "red", label ="Setosa")

plt.scatter(Iris_versicolor["Id"], Iris_versicolor["PetalLengthCm"], color = "green", label ="Versicolor")

plt.scatter(Iris_virginica["Id"], Iris_virginica["PetalLengthCm"], color = "blue", label ="Virginica")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend() # put label to the nice coordination on the graph.

plt.show()
plt.figure(3)

plt.hist(Iris_setosa["PetalLengthCm"], color = "red", label ="Setosa", bins = 25)

plt.hist(Iris_versicolor["PetalLengthCm"], color = "green", label ="Versicolor", bins = 25)

plt.hist(Iris_virginica["PetalLengthCm"], color = "blue", label ="Virginica", bins = 25)

plt.xlabel("PatellengthCm Values")

plt.title("Histogram Graph")

plt.ylabel("Frequency")

plt.legend() # put label to the nice coordination on the graph.

plt.show()
x = np.array([1,2,3,4,5,6,7,8])

y = np.array([2,4,6,8,10,12,14,20])

plt.figure(4)

plt.title("Bar Plot")

plt.xlabel("x")

plt.ylabel("y")

plt.bar(x,y)

plt.show()
# For plotting different plots in the same figure

plt.figure(5) # for plotting graphs in different pages.

plt.title("Sub Plots")

plt.subplot(3,1,1)

plt.plot(Iris_setosa["Id"], Iris_setosa["PetalLengthCm"], color = "red", label ="Setosa")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend()

plt.subplot(3,1,2)

plt.plot(Iris_versicolor["Id"], Iris_versicolor["PetalLengthCm"], color = "green", label ="Versicolor")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend()

plt.subplot(3,1,3)

plt.plot(Iris_virginica["Id"], Iris_virginica["PetalLengthCm"], color = "blue", label ="Virginica")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend() # put label to the nice coordination on the graph.

plt.show()