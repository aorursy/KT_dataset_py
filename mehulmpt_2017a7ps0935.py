# from google.colab import files
# uploader = files.upload()

import pandas as pd

import numpy as np  
data = pd.read_csv('/kaggle/input/dmassign1/data.csv')

data.head()
Data2 = data.replace('?', np.nan)
Data2.isnull().values.any()
Data3 = Data2.fillna({'Class': -1})
Data3.Class.isna().sum()
for i in range(1,189):

  colname = 'Col'+str(i)

  Data3[colname] = Data3[colname].astype('float64') 
Data4 = Data3.fillna(Data3.mean())
for i in range(189, 198):

  colname = 'Col'+str(i)

  Data4[colname] = Data4[colname].fillna(Data4[colname].mode()[0])
DataTrain = Data4.drop('Class', 1)

DataTrain = DataTrain.drop('ID', 1)

DataTrain.head()
for i in range(189,198):

  colname = 'Col'+str(i)

  DataTrain[colname] = pd.Categorical(DataTrain[colname], categories=DataTrain[colname].unique()).codes

# DataTrain['ID'] = pd.Categorical(DataTrain['ID'], categories=DataTrain['ID'].unique()).codes
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



CLUSTERS = 42



ScaledData = StandardScaler().fit(DataTrain).transform(DataTrain)

ScaledDataFrame = pd.DataFrame(ScaledData,columns=DataTrain.columns)

ScaledDataFrame.head()



relation = ScaledDataFrame.corr()

filteredCols = np.full((relation.shape[0]), True, dtype=bool)

for i in range(relation.shape[0]):

    for j in range(i+1, relation.shape[0]):

        if relation.iloc[i,j] > 0.95:

            filteredCols[j] = False



FinalDataFrame = ScaledDataFrame[ScaledDataFrame.columns[filteredCols]]



pca = PCA(n_components=90)

pca.fit(FinalDataFrame)

T1 = pca.transform(FinalDataFrame)
kmeans = KMeans(n_clusters = CLUSTERS)

# kmeans.fit(T1)

y_kmeans = kmeans.fit_predict(T1)
ScaledDataFrame['clusterid'] = y_kmeans
Mapping = {}

for i in range(0,CLUSTERS):

  Mapping[i] = {

      1: 0,

      2: 0,

      3: 0,

      4: 0,

      5: 0

  }



for i in range(0,1300):

  row = ScaledDataFrame.loc[i]

  originalRow = Data4.loc[i]

  classNumber = int(originalRow.Class)

  clusterNumber = int(row.clusterid)



  Mapping[clusterNumber][classNumber] += 1

print(Mapping)
sdic={}

for i in range(0, CLUSTERS):

  cluster = Mapping[i]

  # maxValue = max(cluster[1], cluster[2], cluster[3], cluster[4], cluster[5])

  keyMax = max(cluster, key=cluster.get)

  sdic[i]=keyMax

  print(i, keyMax)
sdic
ScaledDataFrame['Class_PREDICTED'] = ''

for i in range(0, len(ScaledDataFrame)):

  ScaledDataFrame.Class_PREDICTED[i] = sdic[ScaledDataFrame.clusterid[i]]

ScaledDataFrame
# final=ScaledDataFrame.loc[['']]

findf=pd.DataFrame({'ID':(range(1300,13000)),'Class':ScaledDataFrame.Class_PREDICTED[1300:13000]})
findf
for i in range(1300,13000):

  findf['ID'][i]="id"+str(findf['ID'][i])
findf

# findf.to_csv("Attempt3.csv",index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = df.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link(findf)