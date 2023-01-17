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
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
#print(cancer.DESCR)

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df.head()
df.info()
df.describe()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df)
scaled_data = scale.transform(df)
print(type(scaled_data))
from sklearn.decomposition import PCA

pca= PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
type(scaled_data)
x_pca.shape
pd.DataFrame(x_pca)
x_pca[:,1]
plt.scatter(x_pca[:,0], x_pca[:,1], c = cancer["target"], cmap = "plasma")