import numpy as np

import matplotlib.pyplot as plt

import pylab

import seaborn as sb

import pandas as pd

import warnings

warnings.filterwarnings(action="ignore")
# We first read the csv file

iris = pd.read_csv('../input/Iris.csv')

# check the type

type(iris)
# check the content, first 5 lines

iris.head(5)
iris_noid=iris.drop("Id",1)

iris_noid.head(5)
# draw boxplot, we have to use seaborn package instead of matplotlib.pyplot package itself

sb.boxplot(iris_noid)
sb.boxplot(x="SepalLengthCm", y="Species", data=iris_noid)
iris_noid.boxplot(by="Species", column="SepalLengthCm")

iris_noid.boxplot(by="SepalWidthCm", column="SepalLengthCm")

iris_noid.boxplot(by="Species")
sb.violinplot(x="SepalLengthCm", y="Species", data=iris_noid)
sb.pairplot(iris_noid, hue="Species")
pd.tools.plotting.andrews_curves(iris_noid,"Species")