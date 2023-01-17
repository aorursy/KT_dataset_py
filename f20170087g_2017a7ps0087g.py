import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#FileLink = '../input/dataset.csv'

Data = pd.read_csv("/kaggle/input/dmassign1/data.csv")
Data.head()
DataPrep=Data

DataPrep=DataPrep.drop(['Class'],axis=1)

DataPrep=DataPrep.replace({'?':np.nan})

DataPrep=DataPrep.apply(pd.to_numeric,errors="ignore",downcast="float")



DataPrep.info()



DataPrep=DataPrep.fillna(DataPrep.mode())

DataPrep.fillna(value=DataPrep.mode().iloc[0],inplace=True)
#get dummies is done for categorical type data to convert into numerical form

DataPrep = pd.get_dummies(DataPrep, columns=["Col189", "Col190", "Col191", "Col192", "Col193", "Col194", "Col195", "Col196", "Col197"])

final_colums=DataPrep.columns[:]

np.array(final_colums)
#dropping ID column, since we will scale the dataset using zscore normalization

DataPrep = DataPrep.drop(['ID'],axis=1)

from sklearn.preprocessing import StandardScaler

DataPrepFinal=StandardScaler().fit(DataPrep).transform(DataPrep)

DataPrepFinal=pd.DataFrame(DataPrepFinal,columns=DataPrep.columns)

DataPrepFinal.info()
DataPrepFinal.head()

#first 5 rows of completely preprocessed data
NumClusters=100

from sklearn.cluster import AgglomerativeClustering

CLUSTERS = AgglomerativeClustering(n_clusters=NumClusters,affinity='cosine',linkage='average') 

CLUSTERS.fit_predict(DataPrepFinal)

FinalClusterPred=CLUSTERS.labels_
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(DataPrepFinal, "average",metric="cosine")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
FinalClusterPred=pd.DataFrame(FinalClusterPred)

print(FinalClusterPred)
FinalClusterPred.columns = ['Clusters']

#cnt = FinalClusterPred['Clusters'].value_counts() 

#print(cnt)

Dataframe=FinalClusterPred['Clusters']

#df2

preds1=[i for i in Dataframe]

#print(preds1)

len(preds1)
temp=Data.iloc[:1300,198:]

temp
rows=NumClusters

cols =6

arr = [[0 for i in range(cols)] for j in range(rows)]

for i,j in zip(preds1,temp['Class']):

    if(np.isnan(j)):

        break

    else:

        arr[int (i)][int(j)]=arr[int (i)][int(j)]+1



#print(arr)

DFDF = pd.DataFrame(arr)

DFDF=DFDF.T

print(DFDF.to_string())
pred=preds1

d = DFDF 

ans = DFDF.idxmax(axis=0)

for i in range(NumClusters):

    if ans[i] == 0:

        ans[i] = 2 #change to default class instead of 0

 

print(ans.to_string())

Results=pd.DataFrame(np.zeros((13000-1300,2)))

 

for i in range(1300, 13000):

    Results[0][i-1300]='id'+str(i)

    Results[1][i-1300]=int(ans[pred[i]])

 

Results[1]=Results[1].astype(int)



Results.head()
Results.columns = ['ID','Class']

Results.info()

Results.to_csv('2017A7PS0087G_sub.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)

create_download_link(Results)