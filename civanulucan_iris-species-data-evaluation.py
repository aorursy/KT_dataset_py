# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df1 = df.drop(["Id"],axis = 1)

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

print(setosa.describe())
print(versicolor.describe())
print(virginica.describe())
df1 = df.drop(["Id"],axis = 1)
df1.plot(grid = True, alpha = 0.5)
plt.show()
plt.plot(setosa.Id ,setosa.PetalLengthCm, color = "red", label = "Setosa - Petal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalLengthCm")

plt.plot(versicolor.Id ,versicolor.PetalLengthCm, color = "green", label = "versicolor - Petal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalLengthCm")

plt.plot(virginica.Id ,virginica.PetalLengthCm, color = "blue", label = "virginica - Petal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalLengthCm")

plt.plot(setosa.Id ,setosa.PetalWidthCm,'r--', label = "Setosa - Petal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalWidthCm")

plt.plot(versicolor.Id ,versicolor.PetalWidthCm,'g--', label = "versicolor - Petal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalWidthCm")

plt.plot(virginica.Id ,virginica.PetalWidthCm,'b--', label = "virginica - Petal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("PetalWidthCm")
plt.show()

plt.scatter(setosa.PetalLengthCm, setosa.PetalWidthCm, color = "red",  label = "Setosa")
plt.scatter(versicolor.PetalLengthCm, versicolor.PetalWidthCm, color = "green",  label = "versicolor")
plt.scatter(virginica.PetalLengthCm, virginica.PetalWidthCm, color = "blue",  label = "virginica")
plt.legend()
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("Petal Length  - Petal Width Scatter")
plt.show()
plt.hist(setosa.PetalLengthCm,color = "red",label = "Setosa",alpha = 0.5)
plt.hist(versicolor.PetalLengthCm,color = "green",label = "versicolor",alpha = 0.5)
plt.hist(virginica.PetalLengthCm,color = "blue",label = "virginica",alpha = 0.5)

plt.xlabel("PetalLengthCm")
plt.ylabel("Frequency")
plt.title("Petal Length Histogram")

plt.show()

plt.hist(setosa.PetalWidthCm,color = "red",label = "Setosa",alpha = 0.5)
plt.hist(versicolor.PetalWidthCm,color = "green",label = "versicolor",alpha = 0.5)
plt.hist(virginica.PetalWidthCm,color = "blue",label = "virginica",alpha = 0.5)

plt.xlabel("PetalWidthCm")
plt.ylabel("Frequency")
plt.title("Petal Width Histogram")

plt.show()
plt.subplot(3,1,1)
plt.title("PetalLength Comparison")
plt.plot(setosa.Id ,setosa.PetalLengthCm, color = "red", label = "Setosa - Petal Lenght")
plt.ylabel("Setosa")
plt.legend()

plt.subplot(3,1,2)
plt.plot(versicolor.Id ,versicolor.PetalLengthCm, color = "green", label = "versicolor - Petal Lenght")
plt.ylabel("Versicolor")
plt.legend()

plt.subplot(3,1,3)
plt.plot(setosa.Id ,virginica.PetalLengthCm, color = "blue", label = "virginica - Petal Lenght")
plt.ylabel("virginica")
plt.legend()
plt.show()

plt.subplot(3,1,1)
plt.title("PetalWidth Comparison")
plt.plot(setosa.Id ,setosa.PetalWidthCm,'r--', label = "Setosa - Petal Width")
plt.ylabel("Setosa")
plt.legend()

plt.subplot(3,1,2)
plt.plot(versicolor.Id ,versicolor.PetalWidthCm,'g--', label = "versicolor - Petal Width")
plt.ylabel("Versicolor")
plt.legend()

plt.subplot(3,1,3)
plt.plot(setosa.Id ,virginica.PetalWidthCm,'b--', label = "virginica - Petal Width")
plt.ylabel("virginica")
plt.legend()
plt.show()
plt.plot(setosa.Id ,setosa.SepalLengthCm, color = "red", label = "setosa - Sepal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalLengthCm")

plt.plot(versicolor.Id ,versicolor.SepalLengthCm, color = "green", label = "versicolor - Sepal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalLengthCm")

plt.plot(virginica.Id ,virginica.SepalLengthCm, color = "blue", label = "virginica - Sepal Lenght")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalLengthCm")
plt.show()

plt.plot(setosa.Id ,setosa.SepalWidthCm,'r--', label = "setosa - Sepal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalWidthCm")

plt.plot(versicolor.Id ,versicolor.SepalWidthCm,'g--', label = "versicolor - Sepal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalWidthCm")

plt.plot(virginica.Id ,virginica.SepalWidthCm,'b--', label = "virginica - Sepal Width")
plt.legend() 
plt.xlabel("ID")
plt.ylabel("SepalWidthCm")
plt.show()

plt.scatter(setosa.SepalLengthCm, setosa.SepalWidthCm, color = "red",  label = "Setosa")
plt.scatter(versicolor.SepalLengthCm, versicolor.SepalWidthCm, color = "green",  label = "versicolor")
plt.scatter(virginica.SepalLengthCm, virginica.SepalWidthCm, color = "blue",  label = "virginica")
plt.legend()
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.title("Sepal Length  - Sepal Width Scatter")
plt.show()

plt.hist(setosa.SepalLengthCm,color = "red",label = "Setosa",alpha = 0.5)
plt.hist(versicolor.SepalLengthCm,color = "green",label = "versicolor",alpha = 0.5)
plt.hist(virginica.SepalLengthCm,color = "blue",label = "virginica",alpha = 0.5)

plt.xlabel("SepalLengthCm")
plt.ylabel("Frequency")
plt.title("Sepal Length Histogram")

plt.show()

plt.hist(setosa.SepalWidthCm,color = "red",label = "Setosa",alpha = 0.5)
plt.hist(versicolor.SepalWidthCm,color = "green",label = "versicolor",alpha = 0.5)
plt.hist(virginica.SepalWidthCm,color = "blue",label = "virginica",alpha = 0.5)

plt.xlabel("SepalWidthCm")
plt.ylabel("Frequency")
plt.title("Sepal Width Histogram")

plt.show()