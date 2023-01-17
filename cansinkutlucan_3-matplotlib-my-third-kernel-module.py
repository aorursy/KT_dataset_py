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
df = pd.read_csv("../input/iris/Iris.csv")

df
# First take a look the data

df.info()
df.columns
df.Species.unique()
setosa = df[df.Species=="Iris-setosa"]

versicolor = df[df.Species=="Iris-versicolor"]

virginica = df[df.Species=="Iris-virginica"]
setosa.describe()
# Line plot

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label="Setosa")

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="blue",label="Virginica")

plt.plot(virginica.Id,virginica.PetalLengthCm,color="green",label="Versicolor")

plt.title("SPECIES")

plt.xlabel("Id")

plt.ylabel("Petal Length")

plt.legend()

plt.show()
# Lineplot2

df1=df

df1.drop(["Id"],axis=1,inplace=True)

df1.plot(grid=True, alpha=0.8, linestyle=":")

plt.show()
# Scatter Plot

plt.scatter(setosa.Id,setosa.PetalLengthCm)

plt.xlabel("Id")

plt.ylabel("Petal Length")

plt.title("SCATTER")

plt.show()
# Histogram

plt.hist(setosa.PetalLengthCm, bins=10)

plt.show()
# Barplot

countries = ["Russia","U.S.A","Denmark","U.K","Germany","France","Spain"]

expenses = [2000,3000,7000,5000,2500,1500,3500]

plt.bar(countries,expenses)

plt.show()
# Subplot1

df.plot(grid=True, alpha=0.5,subplots=True)

plt.xlabel("Id")

plt.show()
# Subplot 2

plt.subplot(2,1,1)

plt.plot(versicolor.Id , versicolor.PetalLengthCm , color = "green" )



plt.subplot(2,1,2)

plt.plot(virginica.Id , virginica.PetalLengthCm , color = "blue")

plt.xlabel("ID")

plt.ylabel("Petal Length")

plt.show()
# As a data scientist candidate, I wanted to show my new skills on matplotlib

# Thank you for your time.