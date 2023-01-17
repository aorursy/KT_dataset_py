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
df=pd.read_csv("../input/Iris.csv")
# now let make some plots=
# first I want to learn datainfo
df.info()

# Find unique set of species
pd.unique(df["Species"])
# get subdatasets for each 
setosa=df[df["Species"]=="Iris-setosa"]
versicolor=df[df["Species"]=="Iris-versicolor"]
virginica=df[df["Species"]=="Iris-virginica"]



# scatter SepalLengthCm versus SepalWidthCm for each in a figure

plt.scatter(setosa.SepalLengthCm,setosa.SepalWidthCm,color="red",label="setosa")
plt.scatter(versicolor.SepalLengthCm,versicolor.SepalWidthCm,color="green",label="versicolor")
plt.scatter(virginica.SepalLengthCm,virginica.SepalWidthCm,color="yellow",label="virginica")
plt.legend()
plt.title("SepalLengthCm vs SepalWidthCm");
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.grid()





# scatter PetalLengthCm     versus PetalWidthCm for each in a figure

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="yellow",label="virginica")
plt.legend()
plt.title("PetalLengthCm vs PetalWidthCm");
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.grid()