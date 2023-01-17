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
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.head()
df['species'].value_counts()
df.drop('species' , inplace= True , axis =1)
df2 = df.copy() # later Use
df.head()
from sklearn.cluster import KMeans

import pandas as pd

kmeans = KMeans(n_clusters = 3)

kmeans.fit(df)



clust_labels = kmeans.predict(df)



df['flower_type'] = clust_labels
df['flower_type'].value_counts()
df2.head()
df2.drop(['sepal_length' ,'sepal_width' , 'petal_width'] , inplace=True , axis = 1)
from sklearn.cluster import KMeans

import pandas as pd

kmeans = KMeans(n_clusters = 3)

kmeans.fit(df2)



clust_labels = kmeans.predict(df2)

df2['flower_type'] = clust_labels
df2['flower_type'].value_counts()