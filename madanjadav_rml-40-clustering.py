# for basic mathematics operation 

import numpy as np

import pandas as pd

from pandas import plotting



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



# for path

import os

for dirname, _, filenames in os.walk('\kaggle\input'):

    for filename in filenames:

         print(os.path.join(dirname, filename))

#print(os.listdir('kaggle\input'))
#Connecting to the house sales data present in the kaggle input directory.

os.chdir(r'/kaggle/input/telco-customer-churn')

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

data_1 = ff.create_table(data.head())

data.head(10)

desc = ff.create_table(data.describe())

py.iplot(desc)
data.isnull().any().any()
plt.rcParams['figure.figsize'] = (15, 10)





plotting.andrews_curves(data.drop("tenure", axis=1), "gender")

plt.title('Andrew Curves for Gender', fontsize = 20)

plt.show()
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 8)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(data['Annual Income (k$)'])

plt.title('Distribution of Annual Income', fontsize = 20)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')





plt.subplot(1, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(data['Age'], color = 'red')

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Range of Age')

plt.ylabel('Count')

plt.show()
labels = ['Female', 'Male']

size = data['gender'].value_counts()

colors = ['lightgreen', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['tenure'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['MonthlyCharges'], palette = 'rainbow')

plt.title('Distribution of Annual Income', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['TotalCharges'], palette = 'copper')

plt.title('Distribution of Spending Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(),  annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()
#  Gender vs Spendscore



plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(data['gender'], data['TotalCharges'], palette = 'Blues')

plt.title('Gender vs TotalCharges', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(data['gender'], data['MonthlyCharges'], palette = 'rainbow')

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['gender'], data['tenure'], palette = 'Purples', size = 10)

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()


x = data['MonthlyCharges']

y = data['tenure']

z = data['TotalCharges']



sns.lineplot(x, y, color = 'blue')

sns.lineplot(x, z, color = 'pink')

plt.title('Annual Income vs Age and Spending Score', fontsize = 20)

plt.show()
x = data.iloc[:, [3, 4]].values



# let's check the shape of x

print(x.shape)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('TotalCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()

import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('TotalCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
x = data.iloc[:, [2, 4]].values

x.shape
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('TotalCharges')

plt.ylabel('tenure')

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

plt.xlabel('tenure')

plt.ylabel('TotalCharges')

plt.legend()

plt.grid()

plt.show()







x = data[['tenure', 'TotalCharges', 'MonthlyCharges']].values

km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

km.fit(x)

labels = km.labels_

centroids = km.cluster_centers_



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

            xaxis = dict(title  = 'tenure'),

            yaxis = dict(title  = 'MonthlyCharges'),

            zaxis = dict(title  = 'TotalCharges')

        )

)



fig = go.Figure(data = df, layout = layout)

py.iplot(fig)










