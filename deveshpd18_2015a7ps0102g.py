import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import Birch

from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")



import random

random.seed(42)
data = pd.read_csv('../input/dataset/dataset.csv' , low_memory= False)
data.shape
df = data
df.head(5)
df.info()
null_columns = data.columns[data.isnull().any()]
null_columns
#df = df[df.Class.notnull()]

df = df.drop(['class'], 1)
df.info()
df = df.drop_duplicates()
df['monthly period'] = pd.to_numeric( df['monthly period'] , errors= 'coerce')

df['credit1'] = pd.to_numeric( df['credit1'] , errors= 'coerce')

df['installmentrate'] = pd.to_numeric( df['installmentrate'] , errors= 'coerce')

df['tenancy period'] = pd.to_numeric( df['tenancy period'] , errors= 'coerce')

df['installmentcredit'] = pd.to_numeric( df['installmentcredit'] , errors= 'coerce')

df['yearly period'] = pd.to_numeric( df['yearly period'] , errors= 'coerce')

df['age'] = pd.to_numeric( df['age'] , errors= 'coerce')
df=df.fillna(df.mean())
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
df = df.drop(['id','installmentcredit','monthly period','phone'], 1)
df.head(5)
dh = pd.get_dummies(df, columns=['account1','account2','plan','expatriate','gender&type','history','housing','post','sponsors','plotsize','motive','employment period'])

dh.head()
dh.info()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(dh)

dataN = pd.DataFrame(np_scaled)

dataN.head()

dataN.shape

dataN.head()
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']

from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN)

T1 = pca1.transform(dataN)



from sklearn.cluster import KMeans



wcss = []

for i in range(2, 19):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,19),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(16, 8))

kmean = KMeans(n_clusters = 7, random_state = 42)

kmean.fit(dataN)

pred = kmean.predict(dataN)

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
pred
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

    elif pred[i] == 5:

        res.append(1)

    elif pred[i] == 6:

        res.append(1)

    

res
res1 = pd.DataFrame(res)

final = pd.concat([data["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

#final['Class'] = final.Class.astype(int)

final.head()
final=final.drop(final.index[0:175])

final.to_csv('submission.csv', index = False,  float_format='%.f')
from sklearn.neighbors import NearestNeighbors



ns = 62                                                  # If no intuition, keep No. of dim + 1

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

dbscan = DBSCAN(eps=3, min_samples=10)

pred = dbscan.fit_predict(dataN)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters =3 ,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)