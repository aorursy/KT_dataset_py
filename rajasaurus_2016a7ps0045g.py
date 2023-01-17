import sys

!{sys.executable} -m pip install sklearn
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
missing_values = ["?"]

df = pd.read_csv("../input/dataset.csv", na_values = missing_values)

data_orig = df

df
df=df.drop(['id','Class'],axis=1)

print (df.isnull().sum())
df.info()
col_num = [

 "InstallmentRate",

 "Tenancy Period",

 "Age",

 "#Credits",

 "#Authorities",

 "Expatriate",

]



col_str = [

 "Account1",

		"History",

	"Motive",

		"Account2",

	"Employment Period",

		"Gender&Type",

	"Sponsors",

		"Plotsize",

		"Plan",

	"Housing",

		"Post",

		"Phone",

	"Expatriate"

]



col_mean = ["InstallmentCredit",

 "Yearly Period","Monthly Period",

 "Credit1"]
#


#
df.dropna(subset=col_num, inplace=True)

df.dropna(subset=col_str, inplace=True)

df.dropna(subset=col_mean, inplace=True)

print (df.isnull().sum())

#Mukul ad wala tareeka lagao manual 
df.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data_orig[0:175].corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
df = df.astype({"Age": int, "InstallmentCredit": int,"Yearly Period": int})
df = df.astype({"Monthly Period": int, "Credit1": int,"InstallmentRate": int,"Tenancy Period": int})

df=df.drop(['Monthly Period','Credit1'],axis=1)

df.info()
def P_method(v):

    if v=="yes":

        return 1

    else:

        return 0

    

df["Phone"] = df["Phone"].apply(P_method)

df["Phone"].value_counts()
def E_method(v):

    if v:

        return 1

    else:

        return 0

    

df["Expatriate"] = df["Expatriate"].apply(E_method)

df["Expatriate"].value_counts()
def plot_sz(v):

    return v.upper()

df["Plotsize"] = df["Plotsize"].apply(plot_sz)

df
col_str = [

 "Account1",

		"History",

	"Motive",

		"Account2",

	"Employment Period",

		"Gender&Type",

	"Sponsors",

		"Plotsize",

		"Plan",

	"Housing",

		"Post"

]
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
trainData = df

df2 = pd.get_dummies(df, columns=col_str)

#df2=df2.drop(['Expatriate'],axis=1)

df2
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df2)

df2_np = pd.DataFrame(np_scaled)

df2_np
from scipy import stats

zsc = df2.apply(stats.zscore)
zsc.head()
pca2 = PCA(n_components=2)

pca2.fit(df2_np)

T2 = pca2.transform(df2_np)

T3 = pd.DataFrame(T2)

T3.columns=["PCA1", "PCA2"]

T3.plot.scatter(x="PCA1", y="PCA2", marker="o", alpha=0.7)
pca3 = PCA(n_components=2)

pca3.fit(zsc)

T4 = pca2.transform(zsc)

T5 = pd.DataFrame(T2)

T5.columns=["PCA1", "PCA2"]

T5.plot.scatter(x="PCA1", y="PCA2", marker="o", alpha=0.7)
from sklearn.cluster import KMeans

wcss = []

for i in range(2, 6):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(df2_np)

    wcss.append(kmean.inertia_)

    

plt.figure(figsize=(16,10))

plt.plot(range(2,6),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(16, 8))

preds2 = []

for i in range(2, 10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(df2_np)

    pred = kmean.fit_predict(df2_np)

    preds2.append(pred)

    

    plt.subplot(2, 4, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T2[:, 0], T2[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca2.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
plt.figure(figsize=(16, 8))

preds2 = []

for i in range(2, 10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(zsc)

    pred3 = kmean.fit_predict(zsc)

    preds2.append(pred3)

    

    plt.subplot(2, 4, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T4[:, 0], T4[:, 1], c=pred3)

    

    centroids1 = kmean.cluster_centers_

    centroids1 = pca2.transform(centroids1)

    plt.plot(centroids1[:, 0], centroids1[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']
plt.figure(figsize=(16, 8))



kmean = KMeans(n_clusters = 3, random_state = 42, init='k-means++')

kmean.fit(df2_np)

pred = kmean.fit_predict(df2_np)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T2[j,0]

            meany+=T2[j,1]

            plt.scatter(T2[j, 0], T2[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
res = []

for i in range(len(pred)):

    if pred[i] == 0:

        res.append(2)#0

    elif pred[i] == 1:

        res.append(0)#2

    elif pred[i] == 2:

        res.append(1)#1

    

res
temp_df=df2.copy()

temp_df['Class']=res

temp_df.head()

temp2_df=data_orig.copy()

temp2_df=temp2_df.fillna(0)

temp2_df['Class']=temp_df['Class']

temp2_df=temp2_df.fillna(0)

temp2_df['Class']=temp2_df['Class'].astype(int)

temp2_df.head()

temp3=temp2_df[['id','Class']]

temp3
from sklearn.neighbors import NearestNeighbors

ns = 62                                                  

nbrs = NearestNeighbors(n_neighbors = ns).fit(df2_np)

distances, indices = nbrs.kneighbors(df2_np)



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

dbscan = DBSCAN(eps=2, min_samples=10)

pre = dbscan.fit_predict(df2_np)

plt.scatter(T2[:, 0], T2[:, 1], c=pre)
labels1 = dbscan.labels_

#labels1 = labels1[labels1 >= 0]                            #Remove Noise Points

labels1, counts1 = np.unique(labels1, return_counts=True)

print(len(labels1))

print(labels1)

print(len(counts1))

print(counts1)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(df2_np)

plt.scatter(T2[:, 0], T2[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(df2_np, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 3).T
plt.scatter(T2[:,0], T2[:,1], c=y_ac[0,:], s=100, label='')

plt.show()
a=0

for x in range(0,174):

    if data_orig['Class'][x]== temp3['Class'][x]:

        a+=1

    

a
a*100/175
sub1 = temp3[175:]

sub1,len(sub1)
sub1.to_csv('sub14.csv',index=False)
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



create_download_link(sub1)