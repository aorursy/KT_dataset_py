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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.drop(["id","name","host_id","host_name","last_review","room_type"],axis=1,inplace = True)
df.head()
df["reviews_per_month"].fillna(method ='ffill', inplace = True)
df.isnull().sum()
from sklearn.preprocessing  import LabelEncoder 
df['neighbourhood_group']= df[['neighbourhood_group']].apply( LabelEncoder().fit_transform)['neighbourhood_group']
df['neighbourhood']= df[['neighbourhood']].apply( LabelEncoder().fit_transform)['neighbourhood']
df.head()
from sklearn.preprocessing import normalize

# Normalizing Data
nm = normalize(df)
nm = pd.DataFrame(nm, columns=df.columns)
nm = nm.head(n=1000)

# High Value -> Less Price -> More recommended
#nm["price"] = 1 - nm["price"]

# Choosing Columns -> Price & Availability
nm = nm[["price","neighbourhood_group"]]

nm
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
hc = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster = hc.fit_predict(nm)
nm["label"] = cluster

plt.figure(figsize = (15, 10))
plt.scatter(nm["price"][nm.label == 0], nm["neighbourhood_group"][nm.label == 0], color = "green")
plt.scatter(nm["price"][nm.label == 1], nm["neighbourhood_group"][nm.label == 1], color = "blue")
plt.scatter(nm["price"][nm.label == 2], nm["neighbourhood_group"][nm.label == 2], color = "red")
plt.xlabel("price")
plt.ylabel("availability")
plt.show()
import seaborn as sns
sns.lmplot(data=df, x='longitude', y='latitude', hue='calculated_host_listings_count', 
                   fit_reg=False, legend=True, legend_out=True)
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(nm, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Accommodation')
plt.ylabel('Euclidean distances')
plt.show()