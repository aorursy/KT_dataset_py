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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing packages

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

import plotly.graph_objs as go

import plotly as py

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

import sklearn.utils
#Reading the raw data

data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

data.head()
data.set_index('CustomerID', inplace=True)
#print a concise summary of a Data

data.info()
#Return the first 5 rows

data.head()
#get data dimensionality

data.shape
data.describe()
data.isna().sum()
data.duplicated().sum()
data.columns
data.rename(columns={'Annual Income (k$)':'Annual_Income', 'Spending Score (1-100)':'Spending_Score'}, inplace=True)
data.head()
plt.figure(1 , figsize = (15 , 10))

sns.lmplot(x='Age', y='Annual_Income', hue='Gender', data=data)

plt.xlabel('Age'), plt.ylabel('Annual_Income') 

plt.title('Age vs Annual Income w.r.t Gender')

plt.show()
plt.figure(1 , figsize = (15 , 6))

sns.lmplot(x='Age', y='Spending_Score', hue='Gender', data=data)

plt.xlabel('Age'), plt.ylabel('Spending_Score') 

plt.title('Age vs Spending_Score w.r.t Gender')

plt.show()
plt.figure(1 , figsize = (15 , 6))

sns.lmplot(x='Annual_Income', y='Spending_Score', hue='Gender', data=data)

plt.xlabel('Annual_Income'), plt.ylabel('Spending_Score') 

plt.title('Annual_Income vs Spending_Score w.r.t Gender')

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

f.set_figheight(8)

f.set_figwidth(15)

sns.distplot(data['Age'], ax=ax1)

ax1.set_title('Distplot of Age')





sns.distplot(data['Annual_Income'], ax=ax2)

ax2.set_title('Distplot of Annual_Income')



sns.distplot(data['Spending_Score'], ax=ax3)

ax3.set_title('Distplot of Spending_Score')
f, (ax1, ax2, ax3) = plt.subplots(1,3)

f.set_figheight(10)

f.set_figwidth(15)

count = 0

for i in ['Age' , 'Annual_Income' , 'Spending_Score']:

    for j in ['Age' , 'Annual_Income' , 'Spending_Score']:

        if i != j :

            count += 1

            plt.subplot(3 , 3 , count)

            sns.regplot(i , j , data = data)

        

        
plt.figure(1 , figsize = (15 , 6))

ax = sns.boxplot(x="Age", y="Annual_Income", hue="Gender",data=data, palette=sns.color_palette("muted"))
plt.figure(1 , figsize = (15 , 6))

ax = sns.boxplot(x="Age", y="Gender", data=data)
plt.figure(1 , figsize = (15 , 6))

ax = sns.boxplot(x="Annual_Income", y="Gender", data=data)
plt.figure(1 , figsize = (15 , 6))

ax = sns.boxplot(x="Spending_Score", y="Gender", data=data)
df1 = data[['Age', 'Spending_Score']].values

algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")

algorithm.fit(df1)

algorithm.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', algorithm='elkan') )

algorithm.fit(df1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_

y_km = algorithm.fit_predict(df1)

plt.scatter(df1[y_km ==0,0], df1[y_km == 0,1], s=100, c='red')

plt.scatter(df1[y_km ==1,0], df1[y_km == 1,1], s=100, c='black')

plt.scatter(df1[y_km ==2,0], df1[y_km == 2,1], s=100, c='blue')

plt.scatter(df1[y_km ==3,0], df1[y_km == 3,1], s=100, c='cyan')

plt.xlabel('Age')

plt.ylabel('Spending_Score')

plt.title('Segmentation using Age and Spending Score')

df2 = data[['Annual_Income', 'Spending_Score']].values

algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")

algorithm.fit(df2)

algorithm.show()

algorithm = (KMeans(n_clusters = 5 ,init='k-means++', algorithm='elkan') )

algorithm.fit(df2)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_

y_km = algorithm.fit_predict(df2)

plt.scatter(df2[y_km ==0,0], df2[y_km == 0,1], s=100, c='red')

plt.scatter(df2[y_km ==1,0], df2[y_km == 1,1], s=100, c='black')

plt.scatter(df2[y_km ==2,0], df2[y_km == 2,1], s=100, c='blue')

plt.scatter(df2[y_km ==3,0], df2[y_km == 3,1], s=100, c='cyan')

plt.scatter(df2[y_km ==4,0], df2[y_km == 4,1], s=100, c='orange')



plt.xlabel('Annual_Income')

plt.ylabel('Spending_Score')

plt.title('Segmentation using Annual Income and Spending Score')
df3 = data[['Age','Annual_Income', 'Spending_Score']].values

algorithm = KElbowVisualizer(KMeans(init='k-means++',algorithm='elkan'), k=12, metric="distortion")

algorithm.fit(df3)

algorithm.show()

algorithm = (KMeans(n_clusters = 5 ,init='k-means++', algorithm='elkan') )

algorithm.fit(df3)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_

y_km = algorithm.fit_predict(df3)



data['Cluster'] =  labels1

trace1 = go.Scatter3d(

    x= data['Age'],

    y= data['Spending_Score'],

    z= data['Annual_Income'],

    mode='markers',

     marker=dict(

        color = data['Cluster'], 

        size= 30,

        line=dict(

            color= data['Cluster'],

            width= 18

        ),

        opacity=0.8

     )

)

data1 = [trace1]

layout = go.Layout(

#     margin=dict(

#         l=0,

#         r=0,

#         b=0,

#         t=0

#     )

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=data1, layout=layout)

py.offline.iplot(fig)

data.head()
data.describe()
data[data['Cluster']==0].describe()
data[data['Cluster']==1].describe()
data[data['Cluster']==2].describe()
data[data['Cluster']==3].describe()
data[data['Cluster']==4].describe()