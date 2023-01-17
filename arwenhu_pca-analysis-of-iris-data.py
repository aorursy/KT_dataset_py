# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv', header = 0)
#print(iris.head())
pca = PCA(n_components=2) #PCA(n_components='mle') #
pca.fit(iris.drop("Species",axis=1))
print(pca.explained_variance_ratio_) 
iris_r = pca.transform(iris.drop("Species",axis=1))
iris_r = pd.DataFrame(iris_r,columns=["PC1","PC2"])
f = lambda x: round(x,1)
iris_r = iris_r.applymap(f)
print(iris_r.head())
iris_r = pd.concat([iris_r,iris["Species"]],axis=1)
plt.figure(figsize = (10,6))
sns.stripplot(x="PC1",y="PC2",hue="Species",data=iris_r)
plt.xticks(rotation=45)