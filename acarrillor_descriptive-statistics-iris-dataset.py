# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

style.use("seaborn")



import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def in_mas_corp(peso, estatura):

    "Devuelve el indice de masa corporal de la persona. El peso es en kilogramos y la estatura en metros"

    ind=peso/(estatura**2)

    print ("Su índice de masa corporal es: "+str(round(ind,1)))

    return ind
imc_dan=in_mas_corp(82,1.82)

imc_alx=in_mas_corp(70,1.76)
data=pd.read_csv("../input/Iris.csv")

data.set_index("Id",inplace=True)

data.describe()
data.sample(15)
data.var()
# data.loc[:,["PetalWidthCm","SepalWidthCm"]].mean()

print (data["PetalWidthCm"].describe())

print (data["PetalWidthCm"].mode())

print (data["PetalWidthCm"].var())
data["PetalWidthCm"].describe(percentiles=np.arange(0.1,1,0.1))
[n/10 for n in range(1,10,1)]
data["PetalWidthCm"].plot(kind="hist",figsize=(16,9))

# data.hist(figsize=(16,9))
data["PetalWidthCm"].plot(kind="box",figsize=(16,9))
data["PetalWidthCm"].plot(kind="density",figsize=(16,9))
plt.figure(figsize=(16,9))

sns.distplot(data["PetalWidthCm"])
versicolor=data.loc[data.Species=="Iris-versicolor",

         ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

versicolor.sample(10)
plt.figure(figsize=(16,9))

sns.distplot(versicolor.PetalWidthCm)

plt.title("Especie Versicolor")
data.Species.value_counts()