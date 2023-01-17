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
# here is our dataframe:

dframe1 = pd.read_csv("../input/Iris.csv")
dframe1.head()
dframe1.tail()
""" then divide at least three sections and remove the "Id", 
 because we will work on numeric data and we do not want it to interrupt us """

df = dframe1.drop(["Id"], axis = 1)
setosa = dframe1[dframe1.Species == "Iris-setosa"]
versicolor = dframe1[dframe1.Species == "Iris-versicolor"]
virginica = dframe1[dframe1.Species == "Iris-virginica"]
setosa.head(10)
# Let's now take a look via plotting

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")
plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")
plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")
plt.legend()
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.show()
# with grids, our plot is looking better and easier to understand

df.plot(grid=True,alpha= 0.9)
plt.show()
# here are scatter plots.

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")
plt.legend()
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("scatter plot")
plt.show()
# and histograms...

plt.hist(setosa.PetalLengthCm,bins= 10)
plt.xlabel("PetalLengthCm values")
plt.ylabel("frekans")
plt.title("hist")
plt.show()
