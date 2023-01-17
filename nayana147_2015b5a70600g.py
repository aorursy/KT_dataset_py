import sys
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import cluster

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
#import data from csv to dataframe df

f= pd.read_csv("../input/dataset.csv")




f['Sponsors'].unique()
f['Sponsors'].replace({

    'g1':'G1'

    },inplace=True)

f = pd. get_dummies(f, columns=[ 'Sponsors'], prefix = [ 'Sponsors' ])

f.head()
f['Account1'].unique()
f['Account1'].replace({

    'aa':1,

    'ab':2,

    'ac':3,

    'ad':4,

    '?':float('nan')

    },inplace=True)

f.head()
f['History'].unique()
f['History'].replace({

    'c0':1,

    'c1':2,

    'c2':3,

    'c3':4,

    'c4':5,

    '?':float('nan')

    },inplace=True)

f.head()

f['History'].unique()
f['Motive'].unique()
f['Motive'].replace({

    'p0':1,

    'p1':2,

    'p2':3,

    'p3':4,

    'p4':5,

    'p5':6,

    'p6':7,

    'p8':8,

    'p9':9,

    'p10':10,

    '?':float('nan')

    },inplace=True)



f.head()
f['Account2'].unique()
f['Account2'].replace({

    'Sacc4':'sacc4'

    },inplace=True)

f = pd. get_dummies(f, columns=[ 'Account2'], prefix = [ '' ])

f.head()
f['Employment Period'].replace({

    'time1':1,

    'time2':2,

    'time3':3,

    'time4':4,

    'time5':5,

    '?':float('nan')

    },inplace=True)

f.head()
f['Gender&Type'].unique()
f = pd. get_dummies(f, columns=[ 'Gender&Type'], prefix = [ 'Gender&Type' ])

f.head()
f['Plotsize'].unique()
f['Plotsize'].replace({

    'sm':'SM',

    'M.E.':'ME',

    'me':'ME',

    'la':'LA'

    },inplace=True)

f = pd. get_dummies(f, columns=[ 'Plotsize'], prefix = [ 'Plotsize' ])

f.head()
f['Plan'].unique()
f = pd. get_dummies(f, columns=[ 'Plan'], prefix = [ 'Plan' ])

f.head()
f['Housing'].unique()
f = pd. get_dummies(f, columns=[ 'Housing'], prefix = [ 'Housing' ])

f.head()
f_onehot = f. copy()

f_onehot = pd. get_dummies(f_onehot, columns=[ 'Post' ], prefix = [ 'Post' ])



f=f_onehot.copy()

f. head()
f.replace({

    'yes':1,

    'no':0,

    },inplace=True)

f.info()

f.tail()
for c in f.columns[1:]:

    f[c] = f[c].apply(pd.to_numeric, errors="coerce")


f=f.fillna(f.mean())
f.tail()
f.info()




f1=f.drop(columns="id")

f1=f1.drop(columns="Class")

f1=f1.drop_duplicates()
f1.info()


import seaborn as sns

k, ax = plt. subplots(figsize=(10, 8))

corr = f. corr()

#corr is the correlation matrix

sns. heatmap(corr, mask=np. zeros_like(corr, dtype=np. bool), cmap=sns. diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True);
#minmaxscaler

from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

scaled_data=scaler. fit_transform(f1)

scaled_df=pd. DataFrame(scaled_data)

scaled_df. tail()
from sklearn. decomposition import PCA

pca1 = PCA(n_components=2)

pca1. fit(scaled_df)

T1 = pca1. transform(scaled_df)
from sklearn.cluster import KMeans

wcss=[]

for i in range(2,50):

    kmean=KMeans(n_clusters=i,random_state=42)

    kmean.fit(scaled_df)

    wcss.append(kmean.inertia_)

plt. plot(range(2, 50),wcss)

plt. title( ' The Elbow Method ' )

plt. xlabel( ' Number of clusters ' )

plt. ylabel( ' WCSS ' )

plt. show()
colors = [ 'red' , 'green' , 'blue']
plt. figure(figsize=(16, 8))

kmean = KMeans(n_clusters = 3, random_state = 42)

kmean. fit(scaled_df)

pred = kmean. predict(scaled_df)

pred_pd = pd. DataFrame(pred)

arr = pred_pd[0] . unique()

for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T1[j, 0]

            meany+=T1[j, 1]

            plt. scatter(T1[j, 0], T1[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt. annotate(i,(meanx, meany),size=30, weight= 'bold' , color= 'black' , backgroundcolor=colors[i])

pred[0:20]
res = []

for i in range(len(pred)):

    if pred[i] == 1:

        res. append(0)

    elif pred[i] == 0:

        res. append(1)

    elif pred[i] == 2:

        res. append(2)

res
res1 = pd. DataFrame(res)

final = pd. concat([f["id"][175:1031], res1[175:1031]], axis=1) . reindex()

final = final. rename(columns={0: "Class"})

final. head()
final.info()
final.to_csv(' submission5.csv' , index = False)
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