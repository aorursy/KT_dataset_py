# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white", color_codes=True)

%matplotlib inline
iris = pd.read_csv("../input/Iris.csv")

iris.head()
# Counting

sns.countplot(x="PetalWidthCm", data=iris)
iris.SepalWidthCm.hist()
# lmplot

## show the regression models

sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
iris.plot(x ='SepalLengthCm', y = "SepalWidthCm", kind="scatter")
# SNS Jointplot()

## combine scatterpolots and hist in the same figure

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
# FaceGrid

## Use face grid to color or draw plots with many axes

sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
# Hex bin

## Show the dentency and scatter as well.

iris.plot.hexbin(x ='SepalLengthCm', y = "SepalWidthCm", cmap='Oranges', gridsize=25)
# boxPlot

## Used to compare variable change with otheres.

sns.boxplot("Species", y="SepalWidthCm", data=iris)
# Vioin plot 

## It combiine the benefit of boxPlot and stripplot

sns.violinplot("Species", y="SepalWidthCm", data=iris)
#KDEplot

## creates and visualizes a kernel density estimate of the underlying feature

sns.FacetGrid(iris, hue="Species", size=5).map(sns.kdeplot, "SepalLengthCm").add_legend()
# Each variable split by species

iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
# PairPlot

## Shows the bivariate relation between each pair

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3,  diag_kind="kde")
# custermap 

species = iris.pop("Species")

id = iris.pop("Id")



sns.clustermap(iris, cmap="mako")