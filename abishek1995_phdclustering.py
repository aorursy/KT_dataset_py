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
data=pd.read_excel("../input/Train.xlsx")
data.head()
dataframe_suspicious=data[data["Suspicious"]=="Yes"]
dataframe_not_suspicious=data[data["Suspicious"]=='No']
dataframe_indeterminate=data[data['Suspicious']=='indeterminate']
salespersons=data['SalesPersonID'].unique()
temp=pd.DataFrame()
temp['SalesPersonID']=salespersons
temp.head()
temp.shape
temp1=pd.DataFrame(dataframe_suspicious.groupby("SalesPersonID").sum()['TotalSalesValue'])
temp2=pd.DataFrame(dataframe_not_suspicious.groupby("SalesPersonID").sum()['TotalSalesValue'])
temp3=pd.DataFrame(dataframe_indeterminate.groupby("SalesPersonID").sum()['TotalSalesValue'])
temp1=temp1.reset_index()
temp2=temp2.reset_index()
temp3=temp3.reset_index()
iter1=pd.merge(temp,temp1,how='left')
temp2.dtypes
iter2=pd.merge(iter1,temp2,how='left',on='SalesPersonID')
iter3=pd.merge(iter2,temp3,how='left',on='SalesPersonID')
iter3.head()
iter3=iter3.fillna(0)
iter3.shape
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
le=LabelEncoder()
le.fit(iter3['SalesPersonID'])
iter3['SalesPersonID']=le.transform(iter3['SalesPersonID'])
iter3.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(iter3)
print(kmeans.cluster_centers_)
len(kmeans.predict(iter3))
y_km = kmeans.predict(iter3)
value,counts=np.unique(y_km,return_counts=True)

print(np.asarray([value,counts]))
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)

tsne_data = model.fit_transform(iter3)

tsne_data = np.vstack((tsne_data.T, y_km)).T

tsne_data
tsne_data.shape
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "clusterid"))
sns.FacetGrid(tsne_df, hue="clusterid", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()