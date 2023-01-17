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
data_path= '../input/Pokemon.csv'
import matplotlib.pyplot as plt
%matplotlib inline
edata= pd.read_csv(data_path)
edata.head(10)

# Now Trying to cluster the pokemons based on their Total, HP
#first import necessary packages
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler

clus_dataset = edata[['Total','HP']]
clus_dataset = np.nan_to_num(clus_dataset)
clus_dataset = StandardScaler().fit_transform(clus_dataset)

db = DBSCAN(eps=0.15,min_samples=10).fit(clus_dataset) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
edata["clus_db"]= labels

X = edata.values
unique_labels= set(labels)
colours = plt.cm.Spectral(np.linspace(0,1, len(unique_labels)))

for k,col in zip(unique_labels, colours):
    if k==-1:
        col='k'
    class_member_mask = (labels==k)
    
     # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1])

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1])
        
            
