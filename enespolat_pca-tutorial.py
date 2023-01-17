# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import  load_iris

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris= load_iris()
data=iris.data
feature_names=iris.feature_names
y=iris.target

df=pd.DataFrame(data,columns=feature_names)
df["sinif"]=y

x=data
#PCA amaç 4 boyuttan 2 boyuta çekmek
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True)# whiten normalize etmek
pca.fit(x)#4 den 2 ye modelledik
x_pca=pca.transform(x)# transform ile modeli uyguladik

print("variance ratio",pca.explained_variance_ratio_)
#datanın yüzde 97 sine sahibim
print("sum :",sum( pca.explained_variance_ratio_))

#gorsellesme
df.columns
df["p1"]=x_pca[:,0]
df["p2"]=x_pca[:,1]

color=["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif==each],df.p2[df.sinif==each],color=color[each],label=iris.target_names[each])
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
