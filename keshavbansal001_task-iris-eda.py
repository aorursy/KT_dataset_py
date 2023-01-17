# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import csv



iris = pd.read_csv("../input/data.csv")
iris.head()

iris.tail(10)
print("Dataset length: %i\n" % len(iris))
print(iris.shape)

print(iris.columns)
species_list = list(iris["species"].unique())

print(type(iris["species"].unique()))

print(type(species_list))

print("Types of species: %s\n" % species_list)

print(iris['species'].value_counts())


print("Sepal length range: [%s, %s]" % (min(iris["sepal_length"]), max(iris["sepal_length"])))

print("Sepal width range:  [%s, %s]" % (min(iris["sepal_width"]), max(iris["sepal_width"])))

print("Petal length range: [%s, %s]" % (min(iris["petal_length"]), max(iris["petal_length"])))

print("Petal width range:  [%s, %s]\n" % (min(iris["petal_width"]), max(iris["petal_width"])))



print("Sepal length variance:\t %f" % np.var(iris["sepal_length"]))

print("Sepal width variance: \t %f" % np.var(iris["sepal_width"]))

print("Petal length variance:\t %f" % np.var(iris["petal_length"]))

print("Petal width variance: \t %f\n" % np.var(iris["petal_width"]))



print("Sepal length stddev:\t %f" % np.std(iris["sepal_length"]))

print("Sepal width stddev: \t %f" % np.std(iris["sepal_width"]))

print("Petal length stddev:\t %f" % np.std(iris["petal_length"]))

print("Petal width stddev: \t %f\n" % np.std(iris["petal_width"]))
iris.describe()
print(iris["sepal_length"].describe())

print("\n")

print(iris["species"].describe())
iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
sns.set_style("whitegrid") 

sns.FacetGrid(iris, hue="species", height=4).map(plt.scatter, "sepal_length","sepal_width").add_legend()

plt.show()
# pairwise scatter plot: Pair- Plot

# dis-advantages:

## can be used when number of features are high.

## cannot visualize higher dimensional patterns in 3-D and 4-D,

# only possible to view 2D patters.



plt.close()

sns.set_style("whitegrid")

sns.pairplot(iris, hue="species", height=3)

plt.show()
sns.FacetGrid(iris, hue="species", height=5).map(sns.distplot, "petal_length").add_legend() 

plt.show()
iris.hist(

    column=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],

    figsize=(10, 10)

)
