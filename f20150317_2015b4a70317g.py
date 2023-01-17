import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
data_orig = pd.read_csv("../input/dataset.csv", sep=',')

df = data_orig
x = float('nan')

df.replace('?', x, inplace = True)
df = df.drop('id', axis=1)
for column in ['Account1', 'Monthly Period', 'History', 'Motive', 'Credit1', 

               'Account2', 'Employment Period', 'InstallmentRate', 'Gender&Type', 'Sponsors', 'Tenancy Period', 

               'Plotsize', 'Age', 'Plan', 'Housing', 'Post', 'Phone', 'Expatriate', 'InstallmentCredit',

               'Yearly Period']:

    df[column].fillna(df[column].mode()[0], inplace=True)
df.fillna(df.mean(), inplace = True)
df["Motive"].unique()
df = df.drop('Motive', axis=1)
df['Phone'].replace('yes', 1, inplace = True)
df['Phone'].replace('no', 0, inplace = True)
df['Plotsize'].replace('SM', 0, inplace = True)
df['Plotsize'].replace('sm', 0, inplace = True)
df['Plotsize'].replace('me', 1, inplace = True)
df['Plotsize'].replace('ME', 1, inplace = True)
df['Plotsize'].replace('la', 2, inplace = True)
df['Plotsize'].replace('LA', 2, inplace = True)
df['Plotsize'].replace('xl', 3, inplace = True)
df['Plotsize'].replace('M.E.', 2, inplace = True)
df['Plotsize'].replace('XL', 3, inplace = True)
df['Plotsize'].unique()
df1 = df.copy()

df = pd.get_dummies(df, columns=['Account1', 'History', 'Account2', 'Employment Period', 'Gender&Type', 'Sponsors', 'Plan', 'Housing', 'Post'], 

                    prefix = ['Account1', 'History', 'Account2', 'Employment Period', 'Gender&Type', 'Sponsors', 'Plan', 'Housing', 'Post'])
df['Monthly Period'] = df['Monthly Period'].astype(float)

df['Credit1'] = df['Credit1'].astype(float)

df['InstallmentRate'] = df['InstallmentRate'].astype(float)

df['Tenancy Period'] = df['Tenancy Period'].astype(float)

df['Plotsize'] = df['Plotsize'].astype(float)

df['Age'] = df['Age'].astype(float)

df['InstallmentCredit'] = df['InstallmentCredit'].astype(float)

df['Yearly Period'] = df['Yearly Period'].astype(float)
df1 = pd.DataFrame(df)
df.loc[:,"#Credits"].var()
df = df.drop('#Credits', axis=1)
df.loc[:,"#Authorities"].var()
df = df.drop('#Authorities', axis=1)
df.loc[:,"Expatriate"].var()
df = df.drop('Expatriate', axis=1)
df.loc[:, 'Phone'].var()
df = df.drop('Phone', axis=1)
df.info()
df.loc[:, 'InstallmentCredit'].var()
df.loc[:, 'Yearly Period'].var()
df.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(30, 25))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);
df = df.drop('Housing_H3', axis=1)
df = df.drop('Post_Jb1', axis=1)
df = df.drop('Monthly Period', axis=1)
df = df.drop('Yearly Period', axis = 1)
#df.loc[:, 'Monthly Period'].var()
df.loc[:, 'Credit1'].var()
#df.loc[:, 'Housing_H3'].var()
df.loc[:, 'Plotsize'].var()
df.loc[:, 'Employment Period_time1'].var()
#df.loc[:, 'Post_Jb1'].var()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
pca = PCA().fit(dataN1)
plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=30)

pca1.fit(dataN1)

df = pca1.transform(dataN1)
#This algorithm has been performed for comparison. The accuracy measure has been obtained using Agglomerative Clustering



from sklearn.cluster import KMeans



plt.figure(figsize=(30, 16))

preds1 = []



kmean = KMeans(n_clusters = 3, random_state = 26)

kmean.fit(dataN1)

pred = kmean.predict(dataN1)

preds1.append(pred)

    

plt.subplot(2, 5, 2)

plt.title(str(3)+" clusters")

plt.scatter(df[:, 0], df[:, 1], c=pred)

    

centroids = kmean.cluster_centers_

centroids = pca1.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']
#This algorithm has been performed for comparison. The accuracy measure has been obtained using Agglomerative Clustering



plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 3, random_state = 26)

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

            meanx+=df[j,0]

            meany+=df[j,1]

            plt.scatter(df[j, 0], df[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
pred
#This algorithm has been performed for comparison. The accuracy measure has been obtained using Agglomerative Clustering



from sklearn.neighbors import NearestNeighbors



ns = 44                                              # If no intuition, keep No. of dim + 1

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
#This algorithm has been performed for comparison. The accuracy measure has been obtained using Agglomerative Clustering



from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2, min_samples=10)

pred = dbscan.fit_predict(dataN1)

plt.scatter(df[:, 0], df[:, 1], c=pred)
#Run this algorithm to obtain the plot for clustering



from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(dataN1)

plt.scatter(df[:, 0], df[:, 1], c=y_aggclus)
#Run this algorithm to obtain the dendrogram prediction values



from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(dataN1, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
#Run this algorithm to obtain the prediction values



y_ac=cut_tree(linkage_matrix1, n_clusters = 3).T

y_ac
#Run this algorithm to obtain the  plot for the prediction values



plt.scatter(df[:,0], df[:,1], c=y_ac[0,:], s=100, label='')

plt.show()
y_ac = y_ac.tolist()
print(y_ac[0])
res = y_ac[0]
res1 = []

for i in range(len(pred)):

    if res[i] == 0:

        res1.append(0)

    elif res[i] == 1:

        res1.append(1)

    elif res[i] == 2:

        res1.append(2)
res2 = pd.DataFrame(res1)

final = pd.concat([data_orig["id"], res2], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final
final.to_csv('2015B4A70317G.csv', index = False)
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