# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.info()
data.describe()
data.corr()
data.head(7)
data.columns
data.Species.unique()
    #Line Plot
data.PetalWidthCm.plot(color="brown", alpha=0.7, grid=True, figsize=(7,7))
plt.legend()
plt.xlabel("Petal Width Cm")
plt.ylabel("Id")
plt.show()
data1 = data[data.Species == "Iris-setosa"]
data1.describe()
data2 = data[data.Species == "Iris-versicolor"]
data2.describe()
data3 = data[data.Species == "Iris-virginica"]
data3.describe()
    #Line Plot
data1.PetalLengthCm.plot(color="r", alpha=0.7, grid=True, figsize=(7,7))
data2.PetalLengthCm.plot(color="b", alpha=0.7, grid=True, figsize=(7,7))
data3.PetalLengthCm.plot(color="g", alpha=0.7, grid=True, figsize=(7,7))
plt.legend()
plt.xlabel("Petal Length Cm")
plt.ylabel("Id")
plt.show()
    #Scatter Plot
data.plot(kind="scatter", x="PetalWidthCm", y="PetalLengthCm", color="y", figsize=(7,7))
plt.show()
    #Hist Plot
data.PetalLengthCm.plot(kind="hist", color="b", bins=20, figsize=(7,7))
plt.show()
    #Bar Plot
data.plot(kind="bar", x="SepalWidthCm", y="SepalLengthCm", figsize=(7,7))
plt.show()
    #Filtering
data[data.SepalLengthCm > 7]
    #Filtering
data[(data.SepalLengthCm < 5.1) & (data.SepalLengthCm > 4.9)]
    #Dictionary
dictionary = {1:"Adana",
             2:"Adıyaman",
             3:"Afyon",
             4:"Ağrı",
             5:"Amasya",
             6:"Ankara",
             7:"Antalya"}
print(dictionary)
print()
del dictionary[7]    #remove
print(dictionary)
print()
dictionary[7] = "Antalyaa"    #add
print(dictionary)
print()
dictionary[7] = "Antalya"    #update
print(dictionary)
print()
dictionary.clear()
print(dictionary)
print()
del dictionary

    #For Loop
dictionary={"Turkey":"Ankara",
           "USA":"Washington",
           "France":"Paris",
           "Germany":"Berlin",
           "England":"Londra",
           "Italy":"Roma"}

for x,y in dictionary.items():
    print(x," ==> ",y)
    #For Loop
list=[1,2,3,4,5,6,7]

for x,y in enumerate(list):
    print("index : ",x," = ",y)
    #For Loop
for x,y in data[["SepalLengthCm"]][7:21].iterrows():
    print(x,"  ==>  ",y)