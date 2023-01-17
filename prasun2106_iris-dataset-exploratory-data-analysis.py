# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()
iris
iris.keys()
iris.data
iris.target
y = pd.DataFrame(iris.target)
X = pd.DataFrame(iris.data, columns = iris.feature_names)
X.head()
iris.target_names
X.head()
X.info()
X.describe()
X["petal width (cm)"].value_counts()
sns.distplot(X["sepal length (cm)"])
sns.distplot(X["sepal width (cm)"])
sns.distplot(X["petal length (cm)"])
sns.distplot(X["petal width (cm)"])
X.describe()
plt.boxplot(X["sepal length (cm)"])
plt.boxplot(X["petal length (cm)"])
plt.boxplot(X ["petal width (cm)"])
X["petal width (cm)"].unique()
sns.heatmap(X.corr(), cmap  ="Greens" ,annot  = True)
sns.scatterplot(x = "sepal length (cm)", y = "petal length (cm)", hue = "petal width (cm)", data = X)
sns.pairplot(X)
sns.scatterplot(x = "sepal width (cm)", y = "sepal length (cm)", hue = X ["petal length (cm)"], data = X)
sns.scatterplot(x = "sepal width (cm)", y = "sepal length (cm)", hue = X ["petal width (cm)"], data = X)
b = X["petal width (cm)"].unique()
a = X["sepal length (cm)"].unique()
a.shape
b.shape
sns.scatterplot(X["sepal length (cm)"], X["petal length (cm)"])
sns.scatterplot(X["sepal length (cm)"], X["petal length (cm)"], hue = X["sepal width (cm)"])
sns.scatterplot(X["sepal length (cm)"], X["petal length (cm)"], hue = X["petal width (cm)"])
sns.scatterplot(X["sepal length (cm)"], X["petal width (cm)"], hue = X["sepal width (cm)"], palette = "Purples")

sns.scatterplot(X["sepal length (cm)"], X["petal width (cm)"], hue = X["sepal length (cm)"], palette = "Reds")