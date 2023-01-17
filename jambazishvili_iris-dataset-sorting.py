# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output
# show dataframe information

dataframe = pd.read_csv("../input/Iris.csv")

print(dataframe.head(2), end="\n\n")

print(dataframe.info(), end="\n\n")

print(dataframe.describe(), end="\n\n")



# drop column(inplace - in dataframe) Id("Id") column(axis)

dataframe.drop("Id", axis=1, inplace=True)
# plotting Sepal length vs Width

fig1 = dataframe[dataframe.Species=="Iris-setosa"].plot(kind="scatter", color="orange", label="Setosa",

                                                          x="SepalLengthCm", y="SepalWidthCm")

dataframe[dataframe.Species=="Iris-versicolor"].plot(kind="scatter", color="blue", label="Versicolor",

                                                    x="SepalLengthCm", y="SepalWidthCm", ax=fig1)

dataframe[dataframe.Species=="Iris-virginica"].plot(kind="scatter", color="green", label="virginica",

                                                   x="SepalLengthCm", y="SepalWidthCm", ax=fig1)

fig1.set_xlabel("SepalLength")

fig1.set_ylabel("SepalWidth")

fig1.set_title("Sepal Length vs Width")

plt.show()



fig2 = dataframe[dataframe.Species=="Iris-"]


