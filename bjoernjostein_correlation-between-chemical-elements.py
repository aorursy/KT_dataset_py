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
import seaborn as sns

import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as spc

%matplotlib inline

from sklearn import decomposition

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

import scipy
data=pd.read_excel("/kaggle/input/chemical-element-abundances/Chemical element abundances_v2.xls")
data
Chemistry_data = data.iloc[:,2:19]
corrMatrix = Chemistry_data.corr()
plt.figure(figsize=(16,16))

sns.heatmap(corrMatrix, annot=True)

plt.show()
corrDist=scipy.spatial.distance.pdist(corrMatrix, 'correlation')
from scipy.cluster.hierarchy import ward, fcluster
Z=ward(corrDist)
fcluster(Z, 0.9, criterion='distance')
fcluster(Z, 1.1, criterion='distance')
fcluster(Z, 3, criterion='distance')
fcluster(Z, 9, criterion='distance')