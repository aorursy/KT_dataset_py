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
# Dimentionality Reduction Using PCA



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.datasets import load_breast_cancer
lbc = load_breast_cancer()

lbc.keys()
lbc['DESCR']
df = pd.DataFrame(lbc['data'], columns = lbc['feature_names'])

df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



#return type of fit_transform is array

scaled_data = scaler.fit_transform(df)
scaled_data
from sklearn.decomposition import PCA

pca = PCA(n_components=2)



#return type array

x_pca = pca.fit_transform(scaled_data)
x_pca
# Compare shape

print("Scaled Data Shape:")

print(scaled_data.shape)

print("PCA Data Shape:")

print(x_pca.shape)
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0], x_pca[:,1], c=lbc['target'], cmap='plasma')

plt.xlabel("First Principal Component")

plt.ylabel("Second Principal Component")
pca.components_

# Shape of 2x30
df_comp = pd.DataFrame(pca.components_, columns=lbc['feature_names'])

df_comp.head()
plt.figure(figsize=(12,6))

sns.heatmap(df_comp, cmap='plasma')