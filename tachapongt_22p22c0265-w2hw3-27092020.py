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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import cut_tree

import seaborn as sns
Data =pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

Data.head()
Data[Data.isnull().values]
Data[Data.drop(['reviews_per_month','last_review'],axis=1).isnull().values]



Data[Data.drop(['reviews_per_month','last_review','host_name'],axis=1).isnull().values]

Data.fillna({'name':"No_Name"}, inplace=True)#filling null values  with respective values

Data.fillna({'host_name':"No_Name"}, inplace=True)

Data.fillna({'last_review':"Not_Reviewed_Yet"}, inplace=True)

Data.fillna({'reviews_per_month':0}, inplace=True) 

Data.fillna({'last_review':0}, inplace=True) 
Data.isnull().sum()
InputData = Data.drop(['host_id','id','host_name'],axis=1)

InputData
df_data = InputData.groupby('neighbourhood').agg({'price':'mean','minimum_nights':'mean','number_of_reviews': 'mean',

                                                  'calculated_host_listings_count': 'mean','availability_365': 'mean',

                                                 'reviews_per_month':'mean'})

df_data
fig = plt.figure(figsize=(30,30))

mergings = linkage(df_data, method = "ward")

dendrogram(mergings, leaf_font_size=10,labels=df_data.index)

plt.savefig('Dendrograms.png', format='png', bbox_inches='tight')
n =4

hc=AgglomerativeClustering(n_clusters=n, linkage='ward')

hc.fit(df_data)
df_result = df_data.copy()

df_result['cluster']=hc.labels_

df_result.head()
sns.pairplot(df_result,hue='cluster')
for i in range(n):

    a=(df_result['cluster']==i).sum()

    print(a)
df_result[df_result['cluster']==0]
df_result[df_result['cluster']==1]
df_result[df_result['cluster']==2]
df_result[df_result['cluster']==3]