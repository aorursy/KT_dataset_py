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
df=pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')
df.head()
X=df.iloc[:,[3,4]].values
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=4)
model=dbscan.fit(X)
labels=model.labels_
from sklearn import metrics
sample_cores=np.zeros_like(labels,dtype=bool)
#Intialy thw whole is false
sample_cores
sample_cores[dbscan.core_sample_indices_]=True
sample_cores
n_cluster=len(set(labels))-(1 if -1 in labels else 0)
n_cluster
print(metrics.silhouette_score(X,labels))
