# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

print(cancer.DESCR)
print(cancer.data.shape)
from sklearn.decomposition import PCA

scaler= StandardScaler()#instantiate



scaler.fit(cancer.data) # compute the mean and standard which will be used in the next command



X_scaled=scaler.transform(cancer.data)# fit and transform can be applied together and I leave that for simple exercise

# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1



print("after scaling minimum", X_scaled.min(axis=0)) 

pca=PCA(n_components=3) 

pca.fit(X_scaled) 

X_pca=pca.transform(X_scaled) 

#checking the shape of X_pca array

print("shape of X_pca", X_pca.shape)
import matplotlib.pyplot as plt

import pandas as pd

Xax=X_pca[:,0]

Yax=X_pca[:,1]

labels=cancer.target

cdict={0:'red',1:'green'}

labl={0:'Malignant',1:'Benign'}

marker={0:'*',1:'o'}

alpha={0:.3, 1:.5}

fig,ax=plt.subplots(figsize=(7,5))

fig.patch.set_facecolor('white')

for l in np.unique(labels):

 ix=np.where(labels==l)

 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,

           label=labl[l],marker=marker[l],alpha=alpha[l])

# for loop ends

plt.xlabel("First Principal Component",fontsize=14)

plt.ylabel("Second Principal Component",fontsize=14)

plt.legend()

plt.show()