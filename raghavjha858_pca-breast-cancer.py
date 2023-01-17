# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cancer=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
cancer.head(3)
cancer.info(verbose=True)
cancer.describe().T
cancer.isnull().sum()
cancer.head(1)
cancer.diagnosis.value_counts()
cancer.head(1)

drop_list1 = ['id','diagnosis','Unnamed: 32']

can = cancer.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

can.head()
#remove item from the data sets
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(can)
sc=sc.fit_transform(can)
#using pca same as the data sets 

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(sc)

pc = pca.transform(sc)

pc.shape

plt.figure(figsize=(8,6))

plt.scatter(pc[:,0],pc[:,1],cmap='plasma')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')