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
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv("../input/pollution_india_2010 (1).csv",na_values='Null')
data.head()
data.isnull().sum()
data.shape
data=data.dropna()
#Calculate pollution levels state-wise

data_pol=data.groupby('State',as_index=False)[['NO2','PM10','SO2']].agg(np.sum)
data_pol.head()
#Function to scale the data to equilvalent Z-scores

def scale(x):

    return (x-np.mean(x))/np.std(x)

data_num=data_pol.drop("State",axis=1)

data_scaled=data_num.apply(scale,axis=1)
data_scaled.head()
#Alternate to perform scaling using in-built function

from scipy.cluster.hierarchy import dendrogram, linkage
data_scaled=np.array(data_scaled)
Z=linkage(data_scaled,method="ward")
#Plot a Dendogram

fig, ax = plt.subplots(figsize=(15, 20))

ax=dendrogram(Z,orientation="right",labels=np.array(data_pol['State']),leaf_rotation=30,leaf_font_size=16)

plt.tight_layout()

plt.show()