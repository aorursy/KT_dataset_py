import warnings

warnings.filterwarnings("ignore")

import pandas as pd

mallDf = pd.read_csv('../input/Mall_Customers.csv')
mallDf.head()
mallDf.shape
mallDf.all
mallDf.tail(5)
mallDf.columns
mallDf['Gender']
mallDf.head(5)
mallDfCopy = mallDf.copy()
mallDfCopy.head(5)
mallDfCopy['GenderCode'] = mallDfCopy.apply(lambda x: 1 if x['Gender'] == 'Male' else  0, axis = 1)
mallDfCopy.head()
mallDfCopy.describe().transpose()
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # for plot styling

%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')
#Visualizing the data - displot

plot_income = sns.distplot(mallDfCopy["Annual Income (k$)"])

plt.xlabel('Annual Income')
plot_spending_score = sns.distplot(mallDfCopy["Spending Score (1-100)"])

plt.xlabel('Spending Score')
plot_age = sns.distplot(mallDfCopy["Age"])

plt.xlabel('Age')
#Visualizing the data - Violin plot

f, axes = plt.subplots(1,2, figsize=(12,6), sharex=True,sharey=True)

v1 = sns.violinplot(data=mallDfCopy,x='Annual Income (k$)', color="skyblue",ax=axes[0])

v2 = sns.violinplot(data=mallDfCopy,x='Spending Score (1-100)',color="lightgreen",ax=axes[1])

v1.set(xlim=(0,200))
mallDf1 = mallDfCopy[['Annual Income (k$)', 'Spending Score (1-100)']]
#Using the elbow method to find the optium number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

  km = KMeans (n_clusters=i,init='k-means++',max_iter=300, n_init = 10, random_state=0)

  km.fit(mallDf1)

  wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()

##fitting kmeans to the dataset with k=5

km5 = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

y_means = km5.fit_predict(mallDf1)

#Visualizing clusters for k = 5

plt.scatter(mallDf1[y_means==0]['Annual Income (k$)'],mallDf1[y_means==0]['Spending Score (1-100)'],s=50,c='purple',label='Cluster1')

plt.scatter(mallDf1[y_means==1]['Annual Income (k$)'],mallDf1[y_means==1]['Spending Score (1-100)'],s=50,c='blue',label='Cluster2')

plt.scatter(mallDf1[y_means==2]['Annual Income (k$)'],mallDf1[y_means==2]['Spending Score (1-100)'],s=50,c='green',label='Cluster3')

plt.scatter(mallDf1[y_means==3]['Annual Income (k$)'],mallDf1[y_means==3]['Spending Score (1-100)'],s=50,c='cyan',label='Cluster4')

plt.scatter(mallDf1[y_means==4]['Annual Income (k$)'],mallDf1[y_means==4]['Spending Score (1-100)'],s=50,c='yellow',label='Cluster5')



plt.scatter(km5.cluster_centers_[:,0], km5.cluster_centers_[:,1],s=200,marker='s',c='red',alpha=0.7,label='Centroids')

plt.title('Customer Segments')

plt.xlabel('Annual Income of customer')

plt.ylabel('Annual spend from customer on site')

plt.legend()

plt.show()
mallDf1[y_means==0]['Annual Income (k$)']
mallDf2 = mallDfCopy[['Age','Annual Income (k$)','Spending Score (1-100)']]

wcss = []

for i in range(1,11):

  km = KMeans (n_clusters=i,init='k-means++',max_iter=300, n_init = 10, random_state=0)

  km.fit(mallDf2)

  wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
##fitting kmeans to the dataset with k=6

km6 = KMeans(n_clusters=6,init='k-means++',max_iter=300,n_init=10,random_state=0)

km6.fit(mallDf2)
labels = km6.labels_

centroids = km6.cluster_centers_
mallDfCopy['labels'] = labels
import plotly.graph_objs as go

import plotly as py

py.offline.init_notebook_mode(connected = True)



trace1 = go.Scatter3d(

        x = mallDfCopy['Age'],

        y = mallDfCopy['Spending Score (1-100)'],

        z = mallDfCopy['Annual Income (k$)'],

        mode = 'markers',

        marker = dict(

                color = mallDfCopy['labels'],

                size = 20,

                line = dict(

                    color = mallDfCopy['labels'],

                    width = 12

                ),

                opacity = 0.8

        )

)

data = [trace1]

layout = go.Layout(

        title = 'Clusters',

        scene = dict(

                xaxis = dict(title = 'Age'),

                yaxis = dict(title = 'Spending Score'),

                zaxis = dict(title = 'Annual Income')

            )

)

fig = go.Figure(data = data, layout = layout)

py.offline.iplot(fig)