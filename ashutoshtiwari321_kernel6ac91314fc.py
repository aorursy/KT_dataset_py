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
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
data=pd.read_csv('../input/Iris.csv')
data.head()
iris_data=data[data.columns[1:]]
iris_data
species=iris_data['Species'].unique()
print("Total different species involved: %s" % len(species))
iris_data.head()
print("Each type:")
print(iris_data.groupby('Species').size())
summary=iris_data.describe()
res=summary.transpose()
res.head()
sns.set(style="whitegrid", palette="Accent", rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_data)

plt.title("Box plot for Iris",fontsize=20)
plt.show()
sns.set(style="whitegrid", palette="Accent", rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x="Species", y="SepalWidthCm", data=iris_data)

plt.title("Box plot for Iris",fontsize=20)
plt.show()
sns.pairplot(iris_data, hue="Species", palette="BuGn_r",diag_kind="kde")
sns.despine()
plt.show()
sns.pairplot(iris_data, hue="Species", palette="BuGn_r",diag_kind="reg")
sns.despine()
plt.show()
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})
iris2 = pd.melt(iris_data, "Species", var_name="measurement")
sns.swarmplot(x="measurement", y="value", hue="Species",palette="GnBu_d", data=iris2)
sns.despine()
plt.show()
from sklearn.datasets import load_iris
