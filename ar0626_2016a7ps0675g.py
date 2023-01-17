import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
data=pd.read_csv("../input/dataset/dataset.csv")

data_og=data

data.head()
index=data.index

columns=data.columns

values=data.values



columns
data.info()
data=data.replace({False: 0, True: 1})

data=data.replace(to_replace='?', value=None)

dataxyz=data

data['Phone']=data['Phone'].replace({'yes': 1, 'no': 0})

numeric_columns=['Monthly Period','Credit1','InstallmentRate','Tenancy Period','Age','#Credits','#Authorities','InstallmentCredit','Yearly Period']

for i in numeric_columns:

    data[i]=pd.to_numeric(data[i])



data.info()
null_columns=data.columns[data.isnull().any()]

for i in null_columns:

    data[i]=data[i].fillna(data[i].mean())

    

data.columns[data.isnull().any()]
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data=data.drop(['Credit1','id','Monthly Period','Class'],1)

data.info()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



categoric_columns=[]

for i in data.columns:

    if data[i].dtype.name=='object':

            categoric_columns.append(i)



for i in categoric_columns:

    data[i]=data[i].str.lower()

    print(i+' - '+str(len(data[i].unique())))
dataD = pd.get_dummies(data, columns=['Housing','Sponsors','Plan'])

dataD=dataD.drop(['Employment Period','Motive','History','Account1','Account2','Phone','Post','Gender&Type','Plotsize'],1)

dataD.columns
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(dataD)

dataN = pd.DataFrame(np_scaled)

dataN.head()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('Elbow Graph')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(dataN)

T = pca.transform(dataN)
plt.figure(figsize=(16, 8))

preds = []

for i in range(3, 7):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN)

    pred = kmean.predict(dataN)

    preds.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T[:, 0], T[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']

plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 4, random_state = 42)

kmean.fit(dataN)

pred = kmean.predict(dataN)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    

    cl=[0,0,0]

    

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T[j,0]

            meany+=T[j,1]

            plt.scatter(T[j, 0], T[j, 1], c=colors[i])

        

            if j<=175:

                if data_og['Class'][j]==0:

                    cl[0]+=1

                if data_og['Class'][j]==1:

                    cl[1]+=1

                if data_og['Class'][j]==2:

                    cl[2]+=1

                

    net_class=cl.index(max(cl))

                

            

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])

    plt.annotate(net_class,(meanx+0.1, meany+0.1),size=30, weight='bold', color='gray', backgroundcolor=colors[i])
og_ids=data_og['id'].values[175:1031]

final_classes=[]

for i in range(175,1031):

    if pred[i]==1:

        final_classes.append(1)

    if pred[i]==3 or pred[i]==0:

        final_classes.append(2)

    if pred[i]==2:

        final_classes.append(0)



final_results=pd.concat([pd.DataFrame(og_ids),pd.DataFrame(final_classes)], axis=1).reindex()

final_results.columns=['id','Class']

final_results['Class']=final_results['Class'].astype(int)

final_results.head()
final_results.info()
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN)

plt.scatter(T[:, 0], T[:, 1], c=y_aggclus)
from sklearn.neighbors import NearestNeighbors



ns = 18                                                  # If no intuition, keep No. of dim + 1

nbrs = NearestNeighbors(n_neighbors = ns).fit(dataN)

distances, indices = nbrs.kneighbors(dataN)



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

dbscan = DBSCAN(eps=0.7, min_samples=10)

predDB = dbscan.fit_predict(dataN)

plt.scatter(T[:, 0], T[:, 1], c=predDB)
final_results.to_csv('2016A7PS0675G_SUB_MEGA2.csv', index=False)

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



create_download_link(final_results)