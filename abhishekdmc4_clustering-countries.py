import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cluster import KMeans



df=pd.read_csv("../input/country-dataset/Country-data.csv")

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.exports=df.exports*df.gdpp

df.health=df.health*df.gdpp

df.imports=df.imports*df.gdpp
df.head()
num_vars=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
plt.figure(figsize=(15,20))

for i in enumerate(num_vars):

    plt.subplot(5,2, i[0]+1)

    sns.distplot(df[i[1]])
# based on this graph I will use GDPP, Income and Chil_mort for cluster 

# profiling as they have a better distribution than rest of them 
plt.figure(figsize=(15,20))

for i in enumerate(num_vars):

    plt.subplot(5,2, i[0]+1)

    sns.boxplot(df[i[1]])
# these plots show a lot of outliers and outliers are not good when clustering
df.head()
num_vars=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec','total_fer']

plt.figure(figsize=(15,20))

for i in enumerate(num_vars):

    plt.subplot(5,2,i[0]+1)

    sns.scatterplot(x=i[1],y='gdpp',data=df)

    
# bivariate analysis shows that countries with high GDP has very low child mortality

# high GDP also means high export, import,income, life expectancy, good health

# high GDP also means low inflation, low fertility
# Capping the outliers

num_vars=[ 'exports', 'health', 'imports', 'income','inflation', 'life_expec','total_fer','gdpp','child_mort']

for i in num_vars:

    

    q1=df[i].quantile(0.01)

    q4=df[i].quantile(0.99)

    if i == 'child_mort':

        df[i][df[i]<=q1]=q1

        

    else :

        df[i][df[i]>=q4]=q4

        
#univariate

plt.figure(figsize=(15,20))

for i in enumerate(num_vars):

    plt.subplot(5,2, i[0]+1)

    sns.boxplot(df[i[1]])
#bi-variate

num_vars=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec','total_fer']

plt.figure(figsize=(15,20))

for i in enumerate(num_vars):

    plt.subplot(5,2,i[0]+1)

    sns.scatterplot(x=i[1],y='gdpp',data=df)
#Calculating the Hopkins statistic



from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
# performin hopkins stastic 10 times and taking average

HopList=[]

sum1=0

for i in range (0,9):

    temp=hopkins(df.drop('country', axis = 1))

    HopList.append(temp)

for i in HopList:

    sum1=sum1+i

avg=sum1/len(HopList)



print(f"average Hopkin's score is:{avg}")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df1 = scaler.fit_transform(df.drop('country', axis = 1))

df1=pd.DataFrame(df1,columns = df.columns[1:])

df1
# Choose the value of K

# Silhouette score - measure of similarity of data point to its own cluster compared to other cluster.

# Heuristics - a value of -1 indicates data is not similar in cluster whereas 1 indicates data is similar in cluster 



from sklearn.metrics import silhouette_score

ss = []

for k in range(2, 11):

    kmean = KMeans(n_clusters = k).fit(df1)

    ss.append([k, silhouette_score(df1, kmean.labels_)])

temp = pd.DataFrame(ss)

print(temp)

plt.plot(temp[0], temp[1])

plt.grid()

plt.show()
# Elbow curve or Sum of Squared Distances(ssd)

# Heuristic - choose a number of clusters so that adding another cluster doesn't give much better modeling of the data

ssd = []

for k in range(2, 11):

    kmean = KMeans(n_clusters = k).fit(df1)

    ssd.append([k, kmean.inertia_])

    

temp = pd.DataFrame(ssd)

plt.plot(temp[0], temp[1])

plt.grid()

plt.show()
#### based on silhoutte score and elbow curve i will choose K=3

kmean = KMeans(n_clusters = 3,random_state=100).fit(df1)
df_kmean = df.copy()

label  = pd.DataFrame(kmean.labels_, columns= ['label'])

label.head()
df_kmean = pd.concat([df_kmean, label], axis =1)

df_kmean.head()
df_kmean.label.value_counts()
sns.scatterplot(x = 'gdpp', y = 'income', hue = 'label', data = df_kmean, palette = 'Set1')

plt.grid()

plt.show()
sns.scatterplot(x = 'gdpp', y = 'child_mort', hue = 'label', data = df_kmean, palette = 'Set1')

plt.grid()

plt.show()
sns.scatterplot(x = 'income', y = 'child_mort', hue = 'label', data = df_kmean, palette = 'Set1')

plt.grid()

plt.show()
df_kmean.drop('country', axis = 1).groupby('label').mean().plot(kind = 'barh')

plt.xscale("log")

plt.show()
var=['child_mort','income','gdpp','label']

df_kmean[var].groupby('label').mean().plot(kind = 'bar')

plt.yscale("log")
df_kmean[df_kmean['label'] == 1].sort_values(by=['income','gdpp','child_mort'],ascending=[True,True,False]).head(5)
# Scaled data frame

df1.head()
from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
sl=linkage(df1,method='single', metric='euclidean')

plt.figure(figsize=(20,20))

dendrogram(sl)

plt.show()
cl=linkage(df1,method='complete', metric='euclidean')

plt.figure(figsize=(20,20))

dendrogram(cl)

plt.show()
hc_labels=cut_tree(cl,n_clusters=3).reshape(-1,)
df_hier=df.copy()



df_hier['label']=hc_labels

df_hier.head()
df_hier.label.value_counts()
sns.scatterplot(x = 'gdpp', y = 'income', hue = 'label', data = df_hier, palette = 'Set1')

plt.grid()

plt.show()
sns.scatterplot(x = 'gdpp', y = 'child_mort', hue = 'label', data = df_hier, palette = 'Set1')

plt.grid()

plt.show()
sns.scatterplot(x = 'income', y = 'child_mort', hue = 'label', data = df_hier, palette = 'Set1')

plt.grid()

plt.show()
df_hier.drop('country', axis = 1).groupby('label').mean().plot(kind = 'barh')

plt.xscale("log")

plt.show()
var=['child_mort','income','gdpp','label']

df_hier[var].groupby('label').mean().plot(kind = 'bar')

plt.yscale("log")
df_hier[df_hier['label'] == 0].sort_values(by=['income','gdpp','child_mort'],ascending=[True,True,False]).head(5)