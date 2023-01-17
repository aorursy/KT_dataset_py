import numpy as np 

import pandas as pd

import seaborn as sns
%matplotlib inline

import matplotlib.pyplot as plt
lf=pd.read_csv("../input/dataset.csv")

df = lf
df.head()
df.info()

type(df['Monthly Period'][26])
df[df['Monthly Period']=='0']
def convert(x):

    if x=='?':

        return 0

    else:

        return int(x)
def fltconvert(x):

    if x=='?':

        return 0

    else:

        return float(x)
df['Monthly Period']=df['Monthly Period'].apply(convert)
df['Monthly Period'].mean()
def replace(x):

    if x==0:

        return int(df['Monthly Period'].mean())

    else:

        return x
df['Monthly Period']=df['Monthly Period'].apply(replace)
df[df['Credit1']=='0']
df['Credit1']=df['Credit1'].apply(convert)
def replace2(x):

    if x==0:

        return int(df['Credit1'].mean())

    else:

        return x
df['Credit1']=df['Credit1'].apply(replace2)
df[df['InstallmentRate']=='?']
df['InstallmentRate']=df['InstallmentRate'].apply(convert)
df['InstallmentRate'][928]=df['InstallmentRate'].mean()
df[df['Tenancy Period']=='?']
df['Tenancy Period']=df['Tenancy Period'].apply(convert)
df.loc[156,'Tenancy Period']=df['Tenancy Period'].mean()
df['Tenancy Period'][156]=int(df['Tenancy Period'][156])
df[df['Age']=='?']
df['Age']=df['Age'].apply(convert)
df.loc[[1,1005,1015],'Age']=int(df['Age'].mean())
df[df['InstallmentCredit']=='?']
df['InstallmentCredit']=df['InstallmentCredit'].apply(fltconvert)
df.loc[[921,979,1016],'InstallmentCredit']=df['InstallmentCredit'].mean()
df[df['Yearly Period']=='?']
df['Yearly Period']=df['Yearly Period'].apply(fltconvert)

df.loc[[26,820,1002,1028],'Yearly Period']=df['Yearly Period'].mean()
sns.heatmap(df.corr(),annot=True)
df.drop(['Monthly Period','Credit1'],axis=1,inplace=True)
df.head()
Account1=pd.get_dummies(df['Account1'],drop_first=True)
Account1.head()
History=pd.get_dummies(df['History'],drop_first=True)
History.head()
Account2=pd.get_dummies(df['Account2'])
Account2.head()
Account2.drop('Sacc4',axis=1,inplace=True)
EmploymentPeriod=pd.get_dummies(df['Employment Period'])
EmploymentPeriod.head()
Plan=pd.get_dummies(df['Plan'])
Plan.head()
Housing=pd.get_dummies(df['Housing'])
Housing.head()
Sponsors=pd.get_dummies(df['Sponsors'])
Sponsors.head()
Sponsors.drop('g1',axis=1,inplace=True)
Phone=pd.get_dummies(df['Phone'])
Phone.head()
Expatriate=pd.get_dummies(df['Expatriate'])
Expatriate.head()
df.head()
df.drop(['Plan','Housing','Sponsors','Phone','Expatriate','Plotsize'],axis=1,inplace=True)
df.head()
df=pd.concat([df,Plan,Housing,Sponsors,Expatriate],axis=1)
df.drop(['Account1','History','Motive','Account2','Employment Period','Gender&Type','Post'],axis=1,inplace=True)
df.info()
sns.heatmap(df.corr(),cmap='viridis')
id=df[['id']]
id.head()
df.drop('id',axis=1,inplace=True)
Class=df[['Class']]
df.drop('Class',axis=1,inplace=True)
from sklearn.cluster import KMeans

wcss=[]

for i in range(1, 10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(df)

    wcss.append(kmean.inertia_)

    

plt.plot(range(1,10),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)

data5 = pd.DataFrame(np_scaled)

df.info()
from sklearn.decomposition import PCA

pca=PCA(n_components=3)

pca.fit(data5)

DF=pca.transform(data5)

data5.info()
plt.figure(figsize=(10,10))

preds1 = []

i = 3

kmean = KMeans(n_clusters = i, random_state = 42)

kmean.fit(data5)

pred = kmean.predict(data5)

preds1.append(pred)

    

plt.title(str(i)+" clusters")

plt.scatter(DF[:, 0], DF[:, 1], c=pred)

    

centroids = kmean.cluster_centers_

centroids = pca.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

for i in range(175):

    plt.annotate(int(lf['Class'][i]),(DF[i,0],DF[i,1]),size=20,color='red')
colors=['red','green','blue']
plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 3, random_state = 42)

kmean.fit(data5)

pred = kmean.predict(data5)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=DF[j,0]

            meany+=DF[j,1]

            plt.scatter(DF[j, 0], DF[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
DF
len(pred)
res=pd.DataFrame(pred)

final=pd.concat([id,res],axis=1).reindex()

final=final.rename(columns={0:'Class'})
final.head()
f1=final.loc[175:,:]
f1.to_csv('final1.csv',index=False)
from sklearn.neighbors import NearestNeighbors

ns = 3

nbrs = NearestNeighbors(n_neighbors = ns).fit(data5)

distances, indices = nbrs.kneighbors(data5)



kdist = []



for i in distances:

    avg = 0.0

    for j in i:

        avg += j

    avg = avg/(ns-1)

    kdist.append(avg)



kdist = sorted(kdist)

plt.plot(indices[:,0], kdist)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=3)

pred = dbscan.fit_predict(data5)

plt.scatter(DF[:, 0], DF[:, 1], c=pred)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(data5)

plt.scatter(DF[:, 0], DF[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(data5, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
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



create_download_link(final1.csv)