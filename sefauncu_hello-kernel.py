# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
data.columns
data.Species.unique()
data.info()
data.describe()
setosa = data[data.Species == "Iris-setosa"]

versicolor = data[data.Species == "Iris-versicolor"]

virginica = data[data.Species == "Iris-virginica"]
setosa.describe()
versicolor.describe()
virginica.describe()
plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")

plt.legend()

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.show()
data2 = data.drop(["Id"],axis=1)

data2.plot(grid=True,alpha= 0.9)

plt.show()
plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")

plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")

plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")

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
data2.plot(grid=True,alpha= 0.9,subplots = True)

plt.show()
plt.subplot(2,1,1)

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")

plt.ylabel("setosa -PetalLengthCm")

plt.subplot(2,1,2)

plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")

plt.ylabel("versicolor -PetalLengthCm")

plt.show()