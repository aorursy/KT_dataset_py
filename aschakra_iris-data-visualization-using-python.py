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
# Load dataset

iris = pd.read_csv("../input/Iris.csv")

# Remove Id column

del iris["Id"]

iris.head()
iris["Species"].value_counts()
import seaborn as sns

sns.set()



# Scatter plots for the features

sns.pairplot(iris, hue="Species")
# Scatter plots for the features

sns.pairplot(iris, hue="Species", diag_kind="kde")
sns.set(style="whitegrid", palette="muted")



# "Melt" the dataset

iris2 = pd.melt(iris, "Species", var_name="measurement")



# Draw a categorical scatterplot

sns.swarmplot(x="measurement", y="value", hue="Species", data=iris2)
# Violin Plot for Petal Length (most distinguishing feature)

sns.violinplot(x="Species", y="PetalLengthCm", data=iris)
# Violin Plot for Petal Width

sns.violinplot(x="Species", y="PetalWidthCm", data=iris)
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



# Paired Density and Scatterplot Matrix

g = sns.PairGrid(iris, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
sns.boxplot(x="Species", y="PetalLengthCm", hue="Species", data=iris)