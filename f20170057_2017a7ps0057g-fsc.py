import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
mydata= pd.read_csv('../input/data.csv')
mydata=mydata.drop(['ID','Class'],axis=1)

mydata.head()
#can't find any nan columns as instead of NULL we have '?'

nan_columns = mydata.columns[mydata.isnull().any()]

nan_columns
mydata=mydata.replace({"?": None})

nan_columns = mydata.columns[mydata.isnull().any()]

mydata=mydata.replace({None: 1147000000})



nan_columns
#printing datatypes and no. of null values in each column

col_names=np.array(mydata.columns)

x=np.array(mydata.isnull().sum())



for i in range(len(x)):

  print(col_names[i]+"::  "+str(mydata[col_names[i]].dtypes)+"::   "+str(x[i]))
nan_columns=np.array(nan_columns)
#changing datatypes for columns which contain NULL values and are before Col183 to float64

for i in range(len(nan_columns)):

  y=nan_columns[i][3:] #it gives the col no like 183 for Col183

  y=int(str(y))

  if y<189: #as all the columns below 189 number are numeric values

    mydata[nan_columns[i]]=mydata[nan_columns[i]].astype('float64')

#removing NULL values and replacing them with median() values

mydata=mydata.replace({1147000000: None})

for i in range(len(nan_columns)):

  y=nan_columns[i][3:]

  y=int(str(y))

  if y<189:

    mydata["Col"+str(y)].fillna(mydata["Col"+str(y)].median(), inplace=True)
#printing the unique values

for i in range(189,198):

  print("Col"+str(i)+"::  "+str(mydata["Col"+str(i)].unique()))
#dropping all columns other than col189 to reduce dimensions

mydata2=mydata.drop(['Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'],axis=1)

mydata2=pd.get_dummies(mydata2, columns=["Col189"])

mydata2.head()
from sklearn import preprocessing

#Standard Normalization

st_scaler = preprocessing.StandardScaler()

np_scaled = st_scaler.fit_transform(mydata2)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
#performing PCA

from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN1)

T1 = pca1.transform(dataN1)
#testing by elbow method for kmeans

from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#testing by carinski-harabasz score

from sklearn import metrics



preds1 = []

for i in range(2,50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    pred = kmean.predict(dataN1)

    preds1.append(metrics.calinski_harabasz_score(dataN1, kmean.labels_))



    

plt.plot(range(2,50),preds1)

plt.title('The Calinski-Harabasz Index')

plt.xlabel('Number of clusters')

plt.ylabel('Index')

plt.show()
#herarchical clusterirng

from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 21,affinity='cosine',linkage='average',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN1)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(dataN1, "average",metric="cosine")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_transit=cut_tree(linkage_matrix1, n_clusters = 21).T

y_transit
y_transit.reshape(-1,)

y_transit
y_transit.shape
#filling the dictionary which has the index of the pts for every cluster 

numOfClusters=21

cluster_pnts={}

Its_clear={} 

for i in range(numOfClusters):

  cluster_pnts[i]=[]

  Its_clear[i]=0
#trying to confirm that each cluster has atleast one data pt of any of type 1 or 2 or 3 or 4 or 5

for i in range(len(y_transit[0])):

  cluster_pnts[y_transit[0][i]].append(i)

  if i<1300:

    Its_clear[y_transit[0][i]]+=1

#printing the dictionary which is telling the index of data pts every cluster has

cluster_pnts
my_df=pd.read_csv("../input/data.csv")

y_classes=np.array(my_df['Class']).astype(int)
my_ans=np.arange(13000)

for clstr in range(21):

  cluster1=0

  cluster2=0

  cluster3=0

  cluster4=0

  cluster5=0

  for points in range (len(cluster_pnts[clstr])):

    

    temp=cluster_pnts[clstr][points]

    if temp<1300:

      if y_classes[temp]==1:

        cluster1+=1

      elif y_classes[temp]==2:

        cluster2+=1

      elif y_classes[temp]==3:

        cluster3+=1

      elif y_classes[temp]==4:

        cluster4+=1

      else:

        cluster5+=1

  my_cnts=[]

  my_cnts.append(cluster1)

  my_cnts.append(cluster2)

  my_cnts.append(cluster3)

  my_cnts.append(cluster4)

  my_cnts.append(cluster5)



  max_freq=max(my_cnts)



  for points in range (len(cluster_pnts[clstr])):

    temp=cluster_pnts[clstr][points]

    if max_freq==cluster1:

      my_ans[temp]=1

    elif max_freq==cluster2:

      my_ans[temp]=2

    elif max_freq==cluster3:

      my_ans[temp]=3

    elif max_freq==cluster4:

      my_ans[temp]=4

    elif max_freq==cluster5:

      my_ans[temp]=5
my_ans.shape
my_df_final=pd.DataFrame()
my_df_final['ID']=my_df['ID']

my_df_final['Class']=my_ans
my_df_final.head(20)
my_output=my_df_final[1300:]
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html     =     '<a     download="{filename}"     href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(my_output)