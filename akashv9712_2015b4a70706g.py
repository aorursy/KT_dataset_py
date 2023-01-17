import pandas as pd

import numpy as np

import sklearn as sk

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
data = pd.read_csv('../input/dataset.csv')
data.head()
cols = list(data.columns)

cols.remove('Class')

cols.remove('id')

X_proc = data[cols]
X_proc.replace(to_replace='?', value=np.nan, inplace=True)

    
X_proc.fillna(method='bfill', inplace=True)

X_proc.fillna(method='ffill', inplace=True)
X_proc.info()
for i in cols:

    try:

        X_proc[i] = X_proc[i].astype('float32')

    except:

        pass
cat_cols = [i for i in cols if X_proc[i].dtype == 'object']
cat_cols
f, ax = plt.subplots(figsize=(10, 8))

corr = X_proc.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
X_proc = pd.get_dummies(data=X_proc, columns=cat_cols)
from sklearn.preprocessing import StandardScaler

X_proc = pd.DataFrame(StandardScaler().fit_transform(X_proc), columns=X_proc.columns)
X_proc.head()
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



# pca = PCA(n_components=20)

# X_proc = pca.fit_transform(X_proc)

# pcols = ['col' + str(i) for i in range(20)]

# X_proc = pd.DataFrame(data=X_proc, columns=pcols)

X_proc.head()


#PCA for plot

pca1 = PCA(n_components=2)

T1 = pca1.fit_transform(X_proc)

plt.figure(figsize=(16, 8))

preds1 = []

for i in range(1, 11):

    kmean = KMeans(n_clusters = 7, random_state = i)

    kmean.fit(X_proc)

    pred = kmean.predict(X_proc)

    preds1.append(pred)

    

    plt.subplot(2, 5, i)

    plt.title(str(i)+" random seed")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
kmean = KMeans(n_clusters = 7, random_state = 10, n_init=25, max_iter=300, init='k-means++')

colors = ['red','green','blue', 'black', 'orange', 'brown', 'pink']

kmean.fit(X_proc)

pred = kmean.predict(X_proc)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

#         if i == data['Class'][j]:

        if i == pred[j]:

            count+=1

            meanx+=T1[j,0]

            meany+=T1[j,1]

            plt.scatter(T1[j, 0], T1[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
plt.scatter(T1[:, 0], T1[:, 1], c=data['Class'])
pred
import itertools as it

labels = [dict((i, j) for i, j in zip(k, (1, 2, 0, 1, 2, 0, 0)))\

          for k in it.permutations([0, 1, 2, 3, 4, 5, 6])]
ans2 = pd.DataFrame(columns=['id', 'Class'])

ans2['id'] = data['id']

ans2['Class'] = pred
from sklearn.metrics import accuracy_score

max_ = 0

index = 0

for i in labels:

    temp = ans2.copy().truncate(after=174)

    temp['Class'] = temp['Class'].map(i).astype('int32')

    x = (accuracy_score(temp['Class'], data['Class'][:175]))

    if(x > max_):

        max_ = x

        index = i
ans2 = pd.DataFrame(columns=['id', 'Class'])

ans2['id'] = data['id']

ans2['Class'] = pred

# print(max_)

# print(index)

# index = {2: 1, 0: 2, 1: 0, 6: 1, 3: 2, 4: 0, 5: 0}
# print(max_)

print(index)

ans2['Class'] = ans2['Class'].map(index).astype('int32')
accuracy_score(ans2['Class'].truncate(after=174), data['Class'].truncate(after=174))
ans2.shape
final = ans2.truncate(before=175)

print(final.shape)

final.to_csv('final.csv', index=False)
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