import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# load iris data

import os



print(os.listdir("../input"))



df = pd.read_csv("../input/Iris.csv")



setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]

virginica = df[df.Species == "Iris-virginica"]
df1 = df.drop(["Id"],axis=1)



plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")



plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.show()

df1.plot(grid=True,alpha= 0.9)

plt.show()
plt.scatter(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.scatter(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.scatter(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")



plt.legend()

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("scatter plot")

plt.show()
plt.hist(setosa.PetalLengthCm,bins= 50)

plt.xlabel("PetalLengthCm values")

plt.ylabel("frekans")

plt.title("hist")

plt.show()
df1.plot(grid=True,alpha= 0.9,subplots = True)

plt.show()



setosa = df[df.Species == "Iris-setosa"]

versicolor = df[df.Species == "Iris-versicolor"]

virginica = df[df.Species == "Iris-virginica"]



plt.subplot(2,1,1)

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.ylabel("setosa -PetalLengthCm")

plt.subplot(2,1,2)

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.ylabel("versicolor -PetalLengthCm")

plt.show()