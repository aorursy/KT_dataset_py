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
import seaborn as sns
iris=pd.read_csv('../input/Iris.csv')
iris.head()
iris.drop('Id',axis=1,inplace=True)
iris.info()
iris['Species'].value_counts()
sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,size=5)
import matplotlib.pyplot as plt
%matplotlib inline
sns.FacetGrid(iris,hue='Species',size=5)\
.map(plt.scatter,'SepalLengthCm','SepalWidthCm')\
.add_legend()
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
ax=sns.stripplot(x='Species',y='SepalLengthCm',data=iris,jitter=True,edgecolor='gray')
ax=sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
ax=sns.stripplot(x='Species',y='SepalLengthCm',data=iris,jitter=True,edgecolor='gray')
sns.violinplot(x='Species',y='SepalLengthCm',data=iris,size=6)
sns.pairplot(data=iris,kind='scatter')
sns.pairplot(iris,hue='Species')
plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(),annot=True,cmap='summer')
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig = sns.swarmplot(x="Species", y="PetalLengthCm", data=iris)

