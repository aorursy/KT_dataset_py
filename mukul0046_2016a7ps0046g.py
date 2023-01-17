import sys
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
#import sklearn.preprocessing as sk

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
dataf=pd.read_csv("../input/dataset.csv", sep=",")

df = dataf

df
for x in range(0,1031):

    for y in range(0,24):

        if df.iloc[x][y]=='?':

            print(x,y)

df.replace('?',np.NaN)



df['Account1'].replace({'?':'ad'},inplace=True)

df['Monthly Period'].replace({'?':18},inplace=True)

df['History'].replace({'?':'c2'},inplace=True)

df['Motive'].replace({'?':'p3'},inplace=True)

df['Credit1'].replace({'?':3000},inplace=True)

df['InstallmentRate'].replace({'?':4},inplace=True)

df['Tenancy Period'].replace({'?':4},inplace=True)

df['Age'].replace({'?':26},inplace=True)

df['InstallmentCredit'].replace({'?':5.0},inplace=True)

df['Yearly Period'].replace({'?':1},inplace=True)



s = pd.Series(df['Credit1'])

s[979] = 2000

s[1016] = 392

df['Credit1'] = s



t = pd.Series(df['Age'])

t[1] = 30

t[1015] = 27

df['Age'] = t



u = pd.Series(df['InstallmentCredit'])

u[921] = 20

u[1016] = 1.6

df['InstallmentCredit'] = u



v = pd.Series(df['Yearly Period'])

v[26] = 4

v[1002] = 1.5

v[1028] = 0.8

df['Yearly Period'] = v



df.drop(['Class', 'id'],1,inplace=True)
df.head()
#dataf=pd.read_csv("./dataset.csv", sep=",")

#df = dataf

df.drop(["Gender&Type", "Housing", "Plan", "Sponsors"],1,inplace=True)

df.drop(["History", "Account2", "Employment Period", "Motive", "InstallmentRate", "Post", "Age"],1,inplace=True)

df = pd.get_dummies(df, columns = [ "Monthly Period",  "Tenancy Period", "#Credits","Plotsize", "#Authorities", "Account1", "InstallmentCredit", "Phone", "Expatriate" ])

#print(df)

#"Yearly Period", "Credit1"
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN1)

T1 = pca1.transform(dataN1)
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2, 10),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 7):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    pred = kmean.predict(dataN1)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']
plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 5, random_state = 42)

kmean.fit(dataN1)

pred = kmean.predict(dataN1)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T1[j,0]

            meany+=T1[j,1]

            plt.scatter(T1[j, 0], T1[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
from sklearn.neighbors import NearestNeighbors



ns = 80                                                 # If no intuition, keep No. of dim + 1

nbrs = NearestNeighbors(n_neighbors = ns).fit(dataN1)

distances, indices = nbrs.kneighbors(dataN1)



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

dbscan = DBSCAN(eps=2, min_samples=1)

pred = dbscan.fit_predict(dataN1)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
labels1 = dbscan.labels_

#labels1 = labels1[labels1 >= 0]                            #Remove Noise Points

labels1, counts1 = np.unique(labels1, return_counts=True)

print(len(labels1))

print(labels1)

print(len(counts1))

print(counts1)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN1)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(dataN1, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
from sklearn.datasets import load_digits

from sklearn.feature_selection import SelectKBest, chi2
dataf=pd.read_csv("../input/dataset.csv", sep=",")

df = dataf



df.drop(['Class', 'id'],1,inplace=True)



data1 = df



data1 = pd.get_dummies(data1, columns = ["Account1","History", "Motive", "Account2", "Employment Period", "InstallmentRate", "Gender&Type",  "Sponsors", "Tenancy Period", "Plotsize", "Plan", "Housing", "#Credits", "Post", "#Authorities", "Phone", "Expatriate",])

df = data1

df.head()
#df = pd.read_csv("./data.csv", sep=",")





#X.shape



ddf = pd.read_csv("../input/dataset.csv", sep=",")

y = ddf['Class'][:175]

ddf.drop(['id','Class'], 1, inplace=True)



ddf.replace('?',np.NaN)



ddf['Account1'].replace({'?':'ad'},inplace=True)

ddf['Monthly Period'].replace({'?':18},inplace=True)

ddf['History'].replace({'?':'c2'},inplace=True)

ddf['Motive'].replace({'?':'p3'},inplace=True)

ddf['Credit1'].replace({'?':3000},inplace=True)

ddf['InstallmentRate'].replace({'?':4},inplace=True)

ddf['Tenancy Period'].replace({'?':4},inplace=True)

ddf['Age'].replace({'?':26},inplace=True)

ddf['InstallmentCredit'].replace({'?':5.0},inplace=True)

ddf['Yearly Period'].replace({'?':1},inplace=True)



s = pd.Series(ddf['Credit1'])

s[979] = 2000

s[1016] = 392

ddf['Credit1'] = s



t = pd.Series(ddf['Age'])

t[1] = 30

t[1015] = 27

ddf['Age'] = t



u = pd.Series(ddf['InstallmentCredit'])

u[921] = 20

u[1016] = 1.6

ddf['InstallmentCredit'] = u



v = pd.Series(ddf['Yearly Period'])

v[26] = 4

v[1002] = 1.5

v[1028] = 0.8

ddf['Yearly Period'] = v



data1 = ddf

data1 = pd.get_dummies(data1, columns = ["Account1", "History", "Motive", "Account2", "Employment Period", "InstallmentRate", "Gender&Type",  "Sponsors", "Tenancy Period", "Plotsize", "Plan", "Housing", "#Credits", "Post", "#Authorities", "Phone", "Expatriate",])

ddf = data1



from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(ddf)

dataN1 = pd.DataFrame(np_scaled)



X = dataN1[:175]
a = SelectKBest(chi2, k=10)

X_new = a.fit_transform(X,y)
X_new
l = a.transform(dataN1)
dataN2 = l
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN2)

T1 = pca1.transform(dataN2)
from sklearn.cluster import KMeans



wcss = []

for i in range(2,10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN2)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,10),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()



plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2,7):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN2)

    pred = kmean.predict(dataN2)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

    

colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']



plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 5, random_state = 42)

kmean.fit(dataN2)

pred = kmean.predict(dataN2)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T1[j,0]

            meany+=T1[j,1]

            plt.scatter(T1[j, 0], T1[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
ddf = pd.read_csv("../input/dataset.csv", sep=",")

res = []

for i in range(len(pred)):

    if pred[i] == 0:

        res.append(2)

    elif pred[i] == 1:

        res.append(0)

    elif pred[i] == 2:

        res.append(2)

    elif pred[i] == 3:

        res.append(1)

    elif pred[i] == 4:

        res.append(0)



a=0

for x in range(0,175):

    if ddf['Class'][x]==res[x]:

        a+=1

a
res1 = pd.DataFrame(res)

final = pd.concat([ddf["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()
sub8 = final[175:]

sub8.to_csv('submission8.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub8)