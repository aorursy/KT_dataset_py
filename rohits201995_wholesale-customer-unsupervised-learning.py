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
df=pd.read_csv('/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv')
df.head()
import seaborn as sns
sns.heatmap(df.isnull()) #to check for null values
#Applying Standardization
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
df_new=SC.fit_transform(df)
df_1=pd.DataFrame(df_new)
df_1.head()
sns.heatmap(df_1[df_1.corr().index].corr(),annot=True) #Checking correlations
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
clusters=[]
for i in range(1,20):
    model = KMeans(n_clusters=i,
               init='k-means++')

    model.fit(df_1)
    clusters.append(model.inertia_)
    
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 20)), y=clusters, ax=ax,marker='o')
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
plt.show()
model = KMeans(n_clusters=7,
               init='k-means++')

model.fit(df_1)
print(model.inertia_,model.cluster_centers_)
