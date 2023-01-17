## importing libraries



import numpy as np

import pandas as pd

from sklearn.cluster import KMeans

from pandas import plotting



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') #538

%matplotlib inline

import seaborn as sns



import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools



init_notebook_mode(connected = True)

import plotly.figure_factory as ff

!ls ../input/customer-segmentation-tutorial-in-python/
# Reading dataset

data = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

dat = ff.create_table(data.head())

py.iplot(dat)
desc = ff.create_table(data.describe())

py.iplot(desc)
data.isnull().sum()
# Data visualization



plt.rcParams['figure.figsize'] = (15,10)

plotting.andrews_curves(data.drop(['CustomerID'],axis =1), "Gender")

plt.title('Andrews curve for Gender')

plt.show()
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (15,5)



plt.subplot(1,2,1)

sns.set(style ='whitegrid')

sns.distplot(data['Annual Income (k$)'])

plt.title('Distribution of Annual Income')

plt.xlabel('Annual Income')

plt.ylabel('Count')

plt.show()



plt.subplot(1,2,2)

sns.set(style='whitegrid')

sns.distplot(data['Age'], color ='red')

plt.title('Distribution of Age')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
data['Gender'].value_counts()
explode =[0,0.1]

colors=['lightgreen','orange']

labels = ['Female','Male']



plt.pie(data['Gender'].value_counts(), labels = labels, explode = explode, colors= colors, shadow = True, autopct ="%.2f%%")

plt.legend()

plt.title('Gender distribution')

plt.axis('off')

plt.show()
plt.rcParams['figure.figsize'] = (10,5)

sns.countplot(data['Age'], palette ='hsv')

plt.title('Distribution of Age')

plt.show()
plt.rcParams['figure.figsize'] = (10,5)

sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Income')

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['Spending Score (1-100)'], palette = 'copper')

plt.title('Distribution of Spending Score', fontsize = 20)

plt.show()
sns.pairplot(data)

plt.show()
plt.rcParams['figure.figsize'] = (10,5)

sns.heatmap(data.corr(), annot = True, cmap ='Wistia')

plt.title('Heatmap')

plt.show()
## Gender vs Spending scores

plt.rcParams['figure.figsize'] = (10,5)

sns.boxenplot(data['Gender'], data['Spending Score (1-100)'], palette = 'Blues')

plt.title('Gender wise spending distribution')

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(data['Gender'], data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['Gender'], data['Age'], palette = 'Purples', size = 10)

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
sns.lineplot(data['Annual Income (k$)'], data['Age'], color='blue')

sns.lineplot(data['Annual Income (k$)'], data['Spending Score (1-100)'], color='pink')

plt.title('Distribution of Annual Income with Age and spending score')

plt.show()

data.columns
x = data.iloc[:,[3,4]].values

x.shape
# kmeans algorithm



#elbow method to find best number of clusters



wcss =[] # within cluster sum of squares

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()

    

#Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

#‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 

#inertia_ = Sum of squared distances of samples to their closest cluster center.

# Visualizing clusters



km = KMeans(n_clusters =5, init ='k-means++', n_init=10, random_state=10, max_iter =300)

y_pred = km.fit_predict(x)
plt.scatter(x[y_pred==0, 0], x[y_pred==0,1], s=100, c ='pink',label='miser')

plt.scatter(x[y_pred==1, 0], x[y_pred==1,1], s=100, c ='yellow',label='general')

plt.scatter(x[y_pred==2, 0], x[y_pred==2,1], s=100, c ='cyan',label='target')

plt.scatter(x[y_pred==3, 0], x[y_pred==3,1], s=100, c ='magenta',label='spendthrift')

plt.scatter(x[y_pred==4, 0], x[y_pred==4,1], s=100, c ='orange',label='careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Kmeans clustering')

plt.xlabel('Annual Income')

plt.ylabel('Spending score')

plt.legend()

plt.grid()

plt.show()
# Hierarchical clustering

import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distance')

plt.show()



#method=’ward’ uses the Ward variance minimization algorithm.
# Visualizing hierarchical clustering



from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters =5, affinity = 'euclidean', linkage ='ward')

y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, c='pink', label = 'miser')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()

## Clustering based on Age



x = data.iloc[:, [2, 4]].values

x.shape
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.grid()

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')



plt.style.use('fivethirtyeight')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()
x = data[['Age','Spending Score (1-100)','Annual Income (k$)']].values

km = KMeans(n_clusters =5, init='k-means++', n_init=10, random_state=10, max_iter =300)

km.fit(x)

labels =km.labels_

centroids = km.cluster_centers_
entroids = km.cluster_centers_

data['labels'] =  labels

trace1 = go.Scatter3d(

    x= data['Age'],

    y= data['Spending Score (1-100)'],

    z= data['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = data['labels'], 

        size= 10,

        line=dict(

            color= data['labels'],

            width= 12

        ),

        opacity=0.8

     )

)

df = [trace1]



layout = go.Layout(

    title = 'Character vs Gender vs Alive or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)



fig = go.Figure(data = df, layout = layout)

py.iplot(fig)