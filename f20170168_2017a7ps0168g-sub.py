import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data= pd.read_csv('../input/dataset/data.csv')
data=data.drop(['ID','Class'],axis=1)

data.head()
data.info()
null_columns = data.columns[data.isnull().any()]

null_columns
obj_cols=data.columns[data.dtypes=='object']
obj_cols=np.array(obj_cols)

obj_cols
data=data.replace({"?": None})

null_columns = data.columns[data.isnull().any()]

data=data.replace({"?": 2147483647})



null_columns
col_names=np.array(data.columns)

nv=np.array(data.isnull().sum())

for i in range(len(nv)):

  print(col_names[i]+":"+str(nv[i])+":"+str(data[col_names[i]].dtypes))
null_columns=np.array(null_columns)
#changing the data-types for columns where NULL is found

for i in range(len(null_columns)):

  x=null_columns[i][3:]

  x=int(str(x))

  if x<189:

    data[null_columns[i]]=data[null_columns[i]].astype('float64')
#replace NULL values with median value

data=data.replace({2147483647: None})

for i in range(len(null_columns)):

  x=null_columns[i][3:]

  x=int(str(x))

  if x<189:

    data["Col"+str(x)].fillna(data["Col"+str(x)].median(), inplace=True)
data['Col187'][643:]
for i in range(189,198):

  print(str(i)+":"+str(data["Col"+str(i)].unique()))

#search for unique values
sizes=np.array(data['Col197'])

for i in range(len(sizes)):

  if sizes[i]=='XL':

    sizes[i]='xl'

  elif sizes[i]=='ME':

    sizes[i]='me'

  elif sizes[i]=='LA':

    sizes[i]='la'

  elif sizes[i]=='M.E.':

    sizes[i]='me'

  elif sizes[i]==None:

    sizes[i]='la'

  elif sizes[i]=='SM':

    sizes[i]='sm'

data['Col197']=sizes

print(data['Col197'].unique())
#drop the object columns to reduce dimensions

data2=data.drop(['Col190','Col191','Col192','Col193','Col194','Col195','Col196'],axis=1)

data2=pd.get_dummies(data2, columns=["Col189","Col197"])

data2.head()
data2.info()
from sklearn import preprocessing

#Performing Standard Normalization

st_scaler = preprocessing.StandardScaler()

np_scaled = st_scaler.fit_transform(data2)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
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

aggclus = AC(n_clusters = 16,affinity='cosine',linkage='average',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN1)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(dataN1, "average",metric="cosine")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 16).T

y_ac
n_cluster=16

cluster_points={}

has_one={}

for i in range(n_cluster):

  cluster_points[i]=[]

  has_one[i]=0

#cluster_points.shape
for i in range(len(y_ac[0])):

  cluster_points[y_ac[0][i]].append(i)

  if i<1300:

    has_one[y_ac[0][i]]+=1
cluster_points
has_one
df=pd.read_csv("../input/dataset/data.csv")

y_classes=np.array(df['Class']).astype(int)
y_classes[0]
ans=np.arange(13000)

for cls in range(16):

  c1=0

  c2=0

  c3=0

  c4=0

  c5=0

  for points in range (len(cluster_points[cls])):

    

    temp=cluster_points[cls][points]

    if temp<1300:

      if y_classes[temp]==1:

        c1+=1

      elif y_classes[temp]==2:

        c2+=1

      elif y_classes[temp]==3:

        c3+=1

      elif y_classes[temp]==4:

        c4+=1

      else:

        c5+=1

  counts=[]

  counts.append(c1)

  counts.append(c2)

  counts.append(c3)

  counts.append(c4)

  counts.append(c5)

  max_freq=max(counts)

  for points in range (len(cluster_points[cls])):

    temp=cluster_points[cls][points]

    if max_freq==c1:

      ans[temp]=1

    elif max_freq==c2:

      ans[temp]=2

    elif max_freq==c3:

      ans[temp]=3

    elif max_freq==c4:

      ans[temp]=4

    elif max_freq==c5:

      ans[temp]=5
ans.shape
df_final=pd.DataFrame()
df_final['ID']=df['ID']

df_final['Class']=ans
df_final
final=df_final[1300:]
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

create_download_link(final)