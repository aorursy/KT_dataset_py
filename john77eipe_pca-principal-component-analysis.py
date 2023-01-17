# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

pdf = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
pdf.head()
pdf.columns
features = pdf.columns.drop(['diagnosis','id','Unnamed: 32']) 
features.size
for i in range(30):

    a = sns.FacetGrid( pdf, hue = 'diagnosis', aspect=4 )

    a.map(sns.kdeplot, features[i], shade= True )

    #a.set(xlim=(0 , pdf['radius_mean'].max()))

    a.add_legend()
X = pdf.drop(['diagnosis','id','Unnamed: 32'], axis = 1)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()#instantiate

scaler.fit(X) # compute the mean and standard which will be used in the next command

X_scaled=scaler.transform(X)# fit and transform can be applied together and I leave that for simple exercise

# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1
X_scaled
from sklearn.decomposition import PCA

pca=PCA(n_components=3) 

pca.fit(X_scaled) 

X_pca=pca.transform(X_scaled) 

#let's check the shape of X_pca array

X_pca.shape
type(X_pca)
pdf = pd.concat([pdf, pd.DataFrame(X_pca)], axis=1)
pdf.head()
pal = dict(M="seagreen", B="gray")

g = sns.FacetGrid(pdf, hue="diagnosis", palette=pal, height=5)

g.map(plt.scatter, 0, 1, s=50, alpha=.7, linewidth=.5, edgecolor="white")

g.add_legend()