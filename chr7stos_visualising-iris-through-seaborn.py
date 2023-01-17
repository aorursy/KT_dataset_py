# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chardet # use chardet to detect the encoding



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(chardet.detect(check_output(["ls", "../input"])))

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

iris = pd.read_csv('../input/Iris.csv')
iris.head()
sns.set(style="white", color_codes=True)
warnings.filterwarnings("ignore")
iris.plot(kind='scatter',x='SepalWidthCm',y='PetalLengthCm')
g = sns.FacetGrid(iris, col="Species")

g.map(sns.distplot, "SepalWidthCm")
g = sns.FacetGrid(iris, col="Species") # need to recreate the FacetGrid object

g.map(plt.scatter, "SepalWidthCm", "PetalWidthCm")
g = sns.FacetGrid(iris, col="Species") # need to recreate the FacetGrid object

g.map(sns.regplot, "SepalWidthCm", "PetalWidthCm")

plt.xlim(1.5, 6)  

plt.ylim(0, 3)  
f = sns.FacetGrid(iris, hue="SepalLengthCm", size=9)

f.map(plt.scatter, "SepalLengthCm", "SepalWidthCm")

f.add_legend()
sns.boxplot(x="Species",y="PetalWidthCm",data=iris)
sns.boxplot(x="Species",y="SepalWidthCm",data=iris)
sns.boxplot(x="Species",y="PetalLengthCm",data=iris)
sns.boxplot(x="Species",y="SepalLengthCm",data=iris)
ax = sns.stripplot(x="Species", y="SepalLengthCm", data=iris, jitter=True)

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")
target = iris.iloc[:,6]