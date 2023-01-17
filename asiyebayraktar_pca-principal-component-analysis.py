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
from sklearn.datasets import load_iris

import pandas as pd
iris = load_iris()

data = iris.data

feature_names = iris.feature_names

y = iris.target



df = pd.DataFrame(data, columns = feature_names)

df.head()
df["cls"] = y
df.head()
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten= True)  # whiten : normalization

pca.fit(data)

x_pca = pca.transform(data)



print("Variance Ratio : ", pca.explained_variance_ratio_)

print("Sum : ", sum(pca.explained_variance_ratio_))



df["p1"] = x_pca[:,0]

df["p2"] = x_pca[:,1]



color = ["red","green","blue"]



import matplotlib.pyplot as plt



for each in range(3):

    plt.scatter(df.p1[df.cls == each],df.p2[df.cls == each], color = color[each],label = iris.target_names[each])

    

plt.xlabel("P1")

plt.ylabel("P2")

plt.legend()

plt.show()

    


