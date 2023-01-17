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





from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler



# for path

import os

print(os.listdir('../input/'))
# importing the dataset

data = pd.read_csv('../input/Mall_Customers.csv')



data.head()
# describing the data



data.describe()
# checking if there is any NULL data



data.isnull().sum()
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (25, 8)



plt.subplot(1, 3, 1)

sns.set(style = 'whitegrid')

sns.distplot(data['Annual Income (k$)'])

plt.title('Distribution of Annual Income', fontsize = 20)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')





plt.subplot(1, 3, 2)

sns.set(style = 'whitegrid')

sns.distplot(data['Age'], color = 'red')

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Range of Age')

plt.ylabel('Count')





plt.subplot(1, 3, 3)

sns.set(style = 'whitegrid')

sns.distplot(data['Spending Score (1-100)'], color = 'green')

plt.title('Distribution of Spending Score', fontsize = 20)

plt.xlabel('Range of Spending Score')

plt.ylabel('Count')

plt.show()
data['Gender'].value_counts().plot(kind='bar')
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['Age'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Annual Income', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['Spending Score (1-100)'], palette = 'copper')

plt.title('Distribution of Spending Score', fontsize = 20)

plt.show()
#Create a numberic label column for gender:

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Gender_Numeric'] = label_encoder.fit_transform(data['Gender'])
sns.pairplot(data)

plt.title('Pairplot for the Data', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()
#  Gender vs Spendscore



plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(data['Gender'], data['Spending Score (1-100)'], palette = 'Blues')

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(data['Gender'], data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Gender vs Annual Income', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['Gender'], data['Age'], palette = 'Purples', size = 10)

plt.title('Gender vs Age', fontsize = 20)

plt.show()
def run_elbow_method_to_find_optimal_no_of_clusters(x):

    from sklearn.cluster import KMeans



    wcss = []

    for i in range(1, 11):

        km = KMeans(n_clusters = i, init = 'k-means++')

        km.fit(x)

        wcss.append(km.inertia_)



    plt.plot(range(1, 11), wcss)

    plt.title('The Elbow Method', fontsize = 20)

    plt.xlabel('No. of Clusters')

    plt.ylabel('wcss')

    plt.show()


import matplotlib

def fit_predict_and_visualize_in_2d_clusters(kmeans, x):

    ### Takes as input a sklearn kmeans model and a 2d matrix and plots the clusters###

    

    if x.shape[1] != 2:

        raise TypeError("This function only accepts matricies with 2 dimensions")

    

    if isinstance(x, pd.core.frame.DataFrame):

        columns = x.columns.to_list()

        x = x.values

        

    color_iterator = matplotlib.colors.cnames.__iter__()

    

    y_means = kmeans.fit_predict(x)

    

    for i in range(kmeans.get_params()['n_clusters']):

        plt.scatter(x[y_means == i, 0], x[y_means == i, 1], s = 100, c = next(color_iterator), label = f'Segment {i}')

        

    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

    plt.style.use('fivethirtyeight')

    plt.title('K Means Clustering', fontsize = 20)

    if columns:

        plt.xlabel(columns[0])

        plt.ylabel(columns[1])

    plt.legend()

    plt.grid()

    plt.show()
def fit_predict_and_visualize_in_3d_clusters(kmeans, x):

    ### Takes as input a sklearn kmeans model and a 3d matrix and plots the clusters###

    

    if x.shape[1] != 3:

        raise TypeError("This function only accepts matricies with 3 dimensions")

        

    kmeans.fit(x)

    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_

    data = x

    data['labels'] =  labels

    trace1 = go.Scatter3d(

        x= data.iloc[:,0],

        y= data.iloc[:,1],

        z= data.iloc[:,2],

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

        margin=dict(

            l=0,

            r=0,

            b=0,

            t=0  

        ),

        scene = dict(

                xaxis = dict(title  = data.columns[0]),

                yaxis = dict(title  = data.columns[1]),

                zaxis = dict(title  = data.columns[2])

            )

    )



    fig = go.Figure(data = df, layout = layout)

    py.iplot(fig)
# NOTE: It is not good design to take a k-means model and the whole dataframe as

# parameters when we only need the cluster centers and the column names, 

# but we wanted to keep the parameters similar between all visialization functions



def fit_predict_and_visualize_cluster_with_barchart(kmeans, x_dataframe):

    y_means = km.fit_predict(x_dataframe)

    df = pd.DataFrame(kmeans.cluster_centers_, columns= x_dataframe.columns)

    df = pd.DataFrame(data=StandardScaler().fit_transform(X=df), columns=df.columns)

    df.plot.bar(figsize=(12,7))

    plt.legend(bbox_to_anchor=(1,.6))

    plt.axhline(color = 'lightgrey', linestyle = '--')

    plt.xlabel('Cluster')
# TODO: Her velger dere hvilke features som skal være med clusteringen, skriv data.columns for å se alle kolonnene



features = ['Age','Gender_Numeric']

x = data[features]


#Plotter WCSS mot antall cluster

#WCSS = the sum of squares of the distances of each data point in all clusters to their respective centroids

run_elbow_method_to_find_optimal_no_of_clusters(x)
#TODO: Her må dere definere hvor mange clustere K-means skal segmentere i.

km = KMeans(n_clusters = 2, init = 'k-means++')

#2d-funksjonen kan kun brukes til å visualisere to dimensjoner, men kan byttes til 3d om man bruker tre dimensjoner.

fit_predict_and_visualize_in_2d_clusters(km, x)

# fit_predict_and_visualize_in_3d_clusters(km, x) #Kan brukes hvis dere har 3 features/dimensjoner som dere lager cluster basert på
fit_predict_and_visualize_cluster_with_barchart(km, x)