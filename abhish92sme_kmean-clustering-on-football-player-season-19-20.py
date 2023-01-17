# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
df=pd.read_csv("../input/whoscoredcom/football.csv")
df.set_index("R",inplace=True)
df.info()

df.head()
fig = px.scatter(df,x="Assists",y="Goals",color="Player",text="Name",title="Goals Vs Assists")
fig.update_traces(textposition='top center')
fig.show()
fig = px.scatter_3d(df,x="Drb",y="Key Pass",z="Shot per Game",color="Player",text="Name",
                    title="Dribbling PerG VS Key Pass PerG VS Shot per Game")
fig.update_traces(textposition='top center')
figsize=(18,8)
fig.show()

plt.rcParams['figure.figsize'] = (18, 6)

plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['Goals'])
plt.title('Goals Distribution', fontsize = 20)
plt.xlabel('Goals')
plt.ylabel('Count')


plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['Age'], color = 'yellow')
plt.title('Assists Distribution', fontsize = 20)
plt.xlabel('Assists Distribution')
plt.ylabel('Count')
plt.show()
X= df.iloc[:, [8,7]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)     
plt.plot(range(1,11), wcss,)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show() 
kmeansmodel = KMeans(n_clusters=4 , init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)
labels = kmeansmodel.labels_
df["label"]=labels
df.loc[df['label'] == 3, 'Category'] = 'magician'
df.loc[df['label'] == 2, 'Category'] = 'Star Performer'
df.loc[df['label'] == 1, 'Category'] = 'Tornado'
df.loc[df['label'] == 0, 'Category'] = 'contributer'

fig = px.scatter(df,x="Assists",y="Goals",color="Category",text="Name",title="K-mean clustering of Assists VS Goals")
fig.update_traces(textposition='top center')
fig.show()
fig=plt.figure(figsize=(10,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Contributer')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'Yellow', label = 'tornado')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Star Performer')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'red', label = 'Magician')
plt.style.use('fivethirtyeight')
plt.title('Player Cluster', fontsize = 15)
plt.xlabel('Assists')
plt.ylabel('Goals')
plt.legend()