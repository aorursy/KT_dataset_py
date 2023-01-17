import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/dataset.csv" , na_values = '?')
data = data_orig.drop(['Class'], 1)

data = data.fillna(method = 'ffill')

data = data.drop_duplicates()

data = data.drop(['id'], 1)
data.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data_orig.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data = data.drop([

#  'id',

 'Account1',

 'Monthly Period',

 'History',

 'Motive',

 'Credit1',

 'Account2',

 'Employment Period',

 'InstallmentRate',

 'Gender&Type',

#  'Sponsors',

 'Tenancy Period',

#  'Plotsize',

#  'Age',

 'Plan',

#  'Housing',

 '#Credits',

 'Post',

 '#Authorities',

 'Phone',

 'Expatriate',

 'InstallmentCredit',

#  'Yearly Period',

#  'Class'

], 1)
data = pd.get_dummies(data, columns = [

#  'id',

#  'Account1',

#  'Monthly Period',

#  'History',

#  'Motive',

#  'Credit1',

#  'Account2',

#  'Employment Period',

#  'InstallmentRate',

#  'Gender&Type',

 'Sponsors',

#  'Tenancy Period',

 'Plotsize',

#  'Age',

#  'Plan',

 'Housing',

#  '#Credits',

#  'Post',

#  '#Authorities',

#  'Phone',

#  'Expatriate',

#  'InstallmentCredit',

#  'Yearly Period',

#  'Class'

])
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data)

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

    

plt.plot(range(2,10),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(dataN1)

    pred = kmean.predict(dataN1)

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=10, color = 'brown', markeredgewidth = 3)
plt.figure(figsize=(16, 8))

colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']

kmean = KMeans(n_clusters = 3, random_state = 42)

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
res = []

for i in range(len(pred)):

    if pred[i] == 0:

        res.append(1)

    elif pred[i] == 1:

        res.append(2)

    elif pred[i] == 2:

        res.append(0)

res = res[175:]
res1 = pd.DataFrame(res, columns = ['Class'])

final = pd.concat([pd.Series(data_orig["id"][175:].values), res1], axis=1).reindex()

final = final.rename(columns={0: "id"})
final.to_csv('submission7.csv', index = False)
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