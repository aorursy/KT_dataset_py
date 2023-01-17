import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.isnull().sum().sort_values(ascending=False)
df.hist(figsize=(12,5))

plt.show()
sns.distplot(df.Age, color='g')
colors = ["#0101DF", "#DF0101"]

sns.countplot('Gender', data=df, palette=colors)
plt.figure(1, figsize=(12, 5))



n = 0



for x in ['Annual Income (k$)', 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1, 2, n)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    sns.distplot(df[x], bins=20)

    plt.title('Distplot of {}'.format(x))

plt.show()
m = df[df['Gender'] == 'Male']

f = df[df['Gender'] == 'Female']
from plotly.offline import init_notebook_mode,iplot


trace1 = go.Histogram(

    x=m['Spending Score (1-100)'],

    opacity=0.75,

    name='MALE')



trace2 = go.Histogram(

    x=f['Spending Score (1-100)'],

    opacity=0.75,

    name='Female')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='spending score according to gender',

                   xaxis=dict(title='Spending Score'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
sns.scatterplot(x=df['Spending Score (1-100)'], y=df['Age'], data=df, hue='Gender')
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Age'], data=df, hue='Gender')
correlation_matrix = df.corr()

fig = plt.figure(figsize=(12,5))

sns.heatmap(correlation_matrix, vmax=0.8, square=True, annot = True)
x = df.iloc[:, [3,4]].values
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1,11), wcss)

plt.title('THE ELBOW METHOD', fontsize=20)

plt.xlabel('No of clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init = 10, random_state=0)

y_means = km.fit_predict(x)
data = df.copy()

data['Targets'] = y_means
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Targets'])

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')
# PLEASE IF YOU READ THIS KERNEL AND YOU LIKE IT OR LEARNT SOMETHING FROM IT PLEASE DONT FORGET TO UPVOTE