import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset_og = pd.read_csv("../input/dataset/dataset.csv",sep=',')

data = dataset_og

data = data.replace({False: 0,True: 1})

data = data.replace({'yes':1,'no':0})

data = data.replace({'?':None})

data = data.replace({'M.E.':'ME'})

numeric_cols=['Monthly Period','Credit1','InstallmentRate','#Credits','#Authorities','Phone','Expatriate','InstallmentCredit','Yearly Period','Age','Tenancy Period']

for i in numeric_cols:

    data[i]=pd.to_numeric(data[i])
data.head()
data.info()
data = data.drop(['Class'],1)

null_columns = data.columns[data.isnull().any()]

print(null_columns)
for i in numeric_cols:

    data[i] = data[i].fillna((data[i].mean()))

data = data.drop_duplicates()

data.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 10))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
#data2 = data.dropna()

data2 = data

data2.info()
null_columns = data2.columns[data2.isnull().any()]

null_columns
data2 = data2.drop(['Monthly Period','Credit1','id'],1)

data2.info()
object_cols=['Account1','History','Motive','Account2','Employment Period','Gender&Type','Sponsors','Plotsize','Plan','Housing','Post']

for i in object_cols:

    data2[i]=data2[i].str.lower()

    print(str(len(data2[i].unique()))+' '+i)

    print(data2[i].unique())
data2 = data2.drop(['Motive',"Account1",'Account2','History',"Phone",'Employment Period','Plotsize',"Post","Gender&Type"],1)



data2 = pd.get_dummies(data2, columns=["Housing","Plan","Sponsors"])

data2.info()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data2)

dataM = pd.DataFrame(np_scaled)

dataM.head()
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataM)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2, 50),wcss)

plt.title('Elbow')

plt.xlabel('clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=3)

pca1.fit(dataM)

T1 = pca1.transform(dataM)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','gray','black','magenta']
plt.figure(figsize=(10, 10))

f = 8

kmean = KMeans(n_clusters = f, random_state = 42)

kmean.fit(dataM)

pred = kmean.predict(dataM)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()

arr_clusters = []



for i in range(f):

    arr_clusters.append(-1)



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    a=0

    b=0

    c=0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T1[j,0]

            meany+=T1[j,1]

            plt.scatter(T1[j, 0], T1[j, 1], c=colors[i])

            if(j<=175):

                if(dataset_og['Class'][j]==0):

                    a=a+1

                elif(dataset_og['Class'][j]==1):

                    b=b+1

                elif(dataset_og['Class'][j]==2):

                    c=c+1

                    

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])

    maxi = max(a,b,c)

    if(maxi==a):

        plt.annotate(0,(meanx+0.2, meany+0.2),size=30, color='orange', backgroundcolor=colors[i])

        arr_clusters[i]=0

    elif(maxi==b):

        plt.annotate(1,(meanx+0.2, meany+0.2),size=30, color='orange', backgroundcolor=colors[i])

        arr_clusters[i]=1

    elif(maxi==c):

        plt.annotate(2,(meanx+0.2, meany+0.2),size=30, color='orange', backgroundcolor=colors[i])

        arr_clusters[i]=2

for i in range(0,175):

    plt.annotate(str(int(dataset_og['Class'][i])),(T1[i][0],T1[i][1]),size=10,color='cyan',weight='bold')

#plot graph, label clusters and points, assign cluster to classes visually
resKM = []

idarr = []

for i in range(len(pred)):

    if(i<=174):

        lol=0

    else:    

        resKM.append(arr_clusters[pred[i]])

        idarr.append(dataset_og['id'][i])

print(len(resKM))

print(arr_clusters)

#assign clusters to classes
zero=0

one=0

two=0

zeroK=0

oneK=0

twoK=0

numcorr = 0

for i in range(0,174):

    if(resKM[i]==0):

        zero=zero+1

    elif(resKM[i]==1):

        one=one+1

    elif(resKM[i]==2):

        two=two+1

    if(dataset_og['Class'][i]==0):

        zeroK=zeroK+1

    elif(dataset_og['Class'][i]==1):

        oneK=oneK+1

    elif(dataset_og['Class'][i]==2):

        twoK=twoK+1

print(str(zero)+" "+str(zeroK))

print(str(one)+" "+str(oneK))

print(str(two)+" "+str(twoK))

#check numbers assigned to each cluster
res1 = pd.DataFrame(idarr)

res2 = pd.DataFrame(resKM)

final = pd.concat([res1,res2], axis=1).reindex()

final.columns = ['id','Class']

final.head()

final.to_csv('final_sub.csv',index=False)

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