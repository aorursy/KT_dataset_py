# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.info()
print(data.columns)
clear_data = data.drop(["Id"],axis=1)
print(clear_data.columns)
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot= True,linewidths= 0.5,fmt= "0.1f",ax=ax)
setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]

plt.plot(setosa.Id, setosa.SepalWidthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.SepalWidthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.SepalWidthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("SepalWidthCm")
plt.title("SepalWidthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.SepalLengthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.SepalLengthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.SepalLengthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("SepalLengthCm")
plt.title("SepalLengthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.PetalLengthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.PetalLengthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.title("PetalLengthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.PetalWidthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.PetalWidthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.PetalWidthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("PetalWidthCm")
plt.title("PetalWidthCm")
plt.legend()
plt.show()



setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm, color="red",label = "setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color="green",label = "versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color="blue",label = "virginica")

plt.legend
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("Scatter Plot 1")
plt.show()


plt.scatter(setosa.SepalWidthCm,setosa.SepalLengthCm, color="red",label = "setosa")
plt.scatter(versicolor.SepalWidthCm,versicolor.SepalLengthCm, color="green",label = "versicolor")
plt.scatter(virginica.SepalWidthCm,virginica.SepalLengthCm, color="blue",label = "virginica")

plt.legend
plt.xlabel("SepalWidthCm")
plt.ylabel("SepalLengthCm")
plt.title("Scatter Plot 2")
plt.show()
plt.hist(setosa.PetalLengthCm, bins=50) # bins are number of bars
plt.xlabel("PetalLengthCm")
plt.ylabel("Frequency of PetalLengthCm ")
plt.title("Histogram 1")
plt.show()

plt.hist(versicolor.SepalWidthCm, bins=55)
plt.xlabel("SepalWidthCm")
plt.ylabel("Frequency of SepalWidthCm ")
plt.title("Histogram 2")
plt.show()

plt.hist(virginica.SepalLengthCm, bins=80)
plt.xlabel("SepalLengthCm")
plt.ylabel("Frequency of SepalLengthCm ")
plt.title("Histogram 3")
plt.show()




clear_data.plot(grid = True, alpha = 0.9, subplots = True)
plt.show()

setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]

plt.subplot(2,1,1)
plt.plot(setosa.Id, setosa.PetalLengthCm, color="red", label ="setosa")
plt.ylabel("setosa - PetalLengthCm")

plt.subplot(2,1,2)
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color="green", label ="versicolor")
plt.ylabel("versicolor - PetalLengthCm")
plt.show()
labels = 'Setosa', 'Versicolor', 'Virginica'
sizes = [33.3,33.3,33.4]
explode = (0, 0, 0,)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["50 Setosa",
          "50 Versicolor",
          "50 Virginica"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="black"))

ax.legend(wedges, ingredients,
          title="Types of Iris",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("IRIS")

plt.show()