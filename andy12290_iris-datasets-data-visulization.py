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
iris = pd.read_csv("../input/Iris.csv")
iris.head()
import seaborn as sns
iris["Species"].value_counts()
import matplotlib.pyplot as plt

%matplotlib inline

iris.plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(x="SepalLengthCm",data = iris)

plt.xticks(rotation = "vertical")
sns.jointplot(x="SepalLengthCm", y = "SepalWidthCm", data = iris)
sns.FacetGrid(iris,size=5, hue="Species").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

                                  
sns.boxplot(x="Species", y="PetalLengthCm", data = iris)
sns.boxplot(x="Species", y="PetalLengthCm", data = iris)

sns.stripplot(x="Species", y="PetalLengthCm", data = iris)
sns.pairplot(iris.drop("Id", axis=1),hue = "Species",size = 2)

    