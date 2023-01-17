# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        global file_path

        file_path = os.path.join(dirname, filename)

        print(file_path)



# Any results you write to the current directory are saved as output.
df = pd.read_csv(file_path)

df.head()
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
#Let us check whether our data is null or not

df.isnull().any().any()
#No, data is not null. Now we will separate out values after analysing data types

df.dtypes
#We will combine all numerical values and save in a separate data frame

df2 = df[['customerID', 'gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges',  'TotalCharges']]

df2.head()
#Let's see ratio of males and females working so that we can try to predict if this conversion is required.

labels = ['Female', 'Male']

size = df2['gender'].value_counts()

colors = ['pink', 'blue']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
#It is great to see almost equal ratio of working males and females. Let's try to analyse the mapping of various values.

plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(df2.corr(),  annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()
x = df2.iloc[:, [3, 4]].values



# let's check the shape of x

print(x.shape)
#Now we will try to see how many clusters are required for an optimum plot

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
#We will provide 5 clusters to our data

km = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'not eligible')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'medium expenditure')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'green', label = 'incentived in future')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'red', label = 'best-target')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Tenure')

plt.ylabel('Monthly Charges')

plt.legend()

plt.grid()

plt.show()
#Hierarchical clustering to anaylse clusters requirement



import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('Employees')

plt.ylabel('Euclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'not eligible')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'best target')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'medium expenditure')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'general')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'incentived in future')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('Tenure')

plt.ylabel('Monthly Charges')

plt.legend()

plt.grid()

plt.show()
#We will try create our model in 3D representation and hence separate the data and values

x = df2[['tenure', 'MonthlyCharges', 'SeniorCitizen']].values

km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

km.fit(x)

labels = km.labels_

centroids = km.cluster_centers_
df2['labels'] =  labels

trace1 = go.Scatter3d(

    x= df2['tenure'],

    y= df2['MonthlyCharges'],

    z= df2['SeniorCitizen'],

    mode='markers',

     marker=dict(

        color = df2['labels'], 

        size= 10,

        line=dict(

            color= df2['labels'],

            width= 12

        ),

        opacity=0.8

     )

)

df3 = [trace1]



layout = go.Layout(

    title = 'Tenure vs Expenditure rate vs Senior Citizen or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

    scene = dict(

            xaxis = dict(title  = 'Tenure'),

            yaxis = dict(title  = 'Monthly Charges'),

            zaxis = dict(title  = 'Senior Citizen')

        )

)



fig = go.Figure(data = df3, layout = layout)

py.iplot(fig)