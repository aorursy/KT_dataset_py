import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)

%matplotlib inline
print("pandas version: {}".format(pd.__version__))

print("numpy version: {}".format(np.__version__))

print("seaborn version: {}".format(sns.__version__))
customer=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customer.head(10)
customer.describe()
customer.info()
customer.isnull().sum()
sns.set(style = 'ticks')

plt.figure(figsize = (15 , 6))

i = 1

for value in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    plt.subplot(1 , 3 , i)

    i +=1

    sns.distplot(customer[value] , bins = 20, kde=False, color="#FF8C00")

    plt.title('Distribution of {}'.format(value))

plt.tight_layout()
age_male = customer[customer['Gender']=='Male']['Age']

age_female = customer[customer['Gender']=='Female']['Age']



age_bin =range(10,75,5) # Minimum Age is 18, Maximum is 70



# Histogram for Male

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13,6), sharey=True)

sns.distplot(age_male, bins=age_bin, kde=False, color='#3498db',ax=ax1, hist_kws=dict(edgecolor="y", linewidth=1))

ax1.set_title('Males')

ax1.set_xticks(age_bin)

ax1.set_ylim(top=20)

ax1.set_ylabel('Count')

ax1.text(45,18,'Average Age: {}'.format(np.around(age_male.mean(),1)) , fontsize=12)



# Histogram for Female

sns.distplot(age_female, bins=age_bin, kde=False, color='#cc66ff',ax=ax2, hist_kws=dict(edgecolor="y", linewidth=1))

ax2.set_title('Females')

ax2.set_xticks(age_bin)

ax2.set_ylim(top=20)

ax2.set_ylabel('Count')

ax2.text(45,18,'Average Age: {}'.format(np.around(age_female.mean(),1)) , fontsize=12)

plt.show()
inc_male = customer[customer['Gender']=='Male']['Annual Income (k$)']

inc_female = customer[customer['Gender']=='Female']['Annual Income (k$)']



income_bin =range(10,150,10) # Minimum Income is 15K, Maximum is 137K



# Histogram for Male

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13,6), sharey=True)

sns.distplot(inc_male, bins=income_bin, kde=False, color='#3498db',ax=ax1, hist_kws=dict(edgecolor="y", linewidth=1))

ax1.set_title('Males')

ax1.set_xticks(income_bin)

ax1.set_ylim(auto=True)

ax1.set_ylabel('Count')

ax1.text(80,19,'Average Income: {}k$'.format(np.around(inc_male.mean(),1)) , fontsize=12)



# Histogram for Female

sns.distplot(inc_female, bins=income_bin, kde=False, color='#cc66ff',ax=ax2, hist_kws=dict(edgecolor="y", linewidth=1))

ax2.set_title('Females')

ax2.set_xticks(income_bin)

ax2.set_ylim(auto=True)

ax2.set_ylabel('Count')

ax2.text(80,19,'Average Income: {}k$'.format(np.around(inc_female.mean(),1)) , fontsize=12)

plt.show()
spscore_male = customer[customer['Gender']=='Male']['Spending Score (1-100)']

spscore_female = customer[customer['Gender']=='Female']['Spending Score (1-100)']



spscore_bin =range(0,105,5) # Minimum Spending Score is 1, Maximum is 99



# Histogram for Male

sns.set_style(style='ticks', rc=None)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5), sharey=True)

sns.distplot(spscore_male, bins=income_bin, kde=False, color='#3498db',ax=ax1, hist_kws=dict(edgecolor="y", linewidth=1))

ax1.set_title('Males')

ax1.set_xticks(spscore_bin)

ax1.set_yticks(range(0,17,1))

ax1.set_xlim(1,100)

ax1.set_ylim(0,16)

ax1.set_ylabel('Count')

ax1.text(40,19,'Average Spending Score: {}'.format(np.around(spscore_male.mean(),1)) , fontsize=12)



# Histogram for Female

sns.distplot(spscore_female, bins=income_bin, kde=False, color='#cc66ff',ax=ax2, hist_kws=dict(edgecolor="y", linewidth=1))

ax2.set_title('Females')

ax2.set_xticks(spscore_bin)

ax2.set_yticks(range(0,17,1))

ax2.set_xlim(1,100)

ax2.set_ylim(0,16)

ax2.set_ylabel('Count')

ax2.text(40,19,'Average Spending Score: {}'.format(np.around(spscore_female.mean(),1)) , fontsize=12)

plt.show()
sns.pairplot(data=customer, hue='Gender')
sns.heatmap(customer.corr(), cmap='coolwarm')
sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=customer, kind='reg')
sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=customer, hue='Gender',aspect=1.5)
sns.lmplot(x='Age', y='Spending Score (1-100)', data=customer, hue='Gender',aspect=1.5)
sns.lmplot(x='Age', y='Annual Income (k$)', data=customer, hue='Gender', aspect=1.5)
cust = customer.drop(['CustomerID','Gender'], axis=1)

cust.head()
from sklearn.cluster import KMeans

find_cls=[]

for i in range(1,15):

    kmean = KMeans(n_clusters=i)

    kmean.fit(cust)

    find_cls.append(kmean.inertia_)

    

   
find_cls
fig, axs = plt.subplots(figsize=(12,5))

sns.lineplot(range(1,15),find_cls, ax=axs, marker='o' )

axs.axvline(5, ls="--", c="crimson")

axs.axvline(6, ls="--", c="crimson")

plt.grid()

plt.show()
kmean_5cls = KMeans(n_clusters=5,init='k-means++')

kmean_5cls.fit(cust)

kmean_5cls.inertia_
kmean_5cls.cluster_centers_
kmean_5cls.labels_
cust['cls_label']=kmean_5cls.labels_
cust.head()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,5),sharey=True)



sns.scatterplot('Annual Income (k$)','Spending Score (1-100)', data=cust, hue='cls_label',

                ax=ax1,palette='Set1', legend='full')

ax1.set_title('Annual Income VS Spending Score (1-100)')

ax1.scatter(kmean_5cls.cluster_centers_[:,1], 

            kmean_5cls.cluster_centers_[:,2], marker='D',c='Black') #Centre for Annual Income & Spending Score

sns.scatterplot('Age','Spending Score (1-100)', data=cust, hue='cls_label', 

                ax=ax2, palette='Set1', legend='full')

ax2.set_title('Age VS Spending Score (1-100)')

ax2.scatter(kmean_5cls.cluster_centers_[:,0], 

            kmean_5cls.cluster_centers_[:,2], marker='D',c='Black') #Centre for Age & Spending Score

plt.show()
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(7, 7))

ax = Axes3D(fig, rect=[0, 0, .99, 1], elev=30, azim=-60)

ax.scatter(cust['Age'],

           cust['Annual Income (k$)'],

           cust['Spending Score (1-100)'],

           c=cust['cls_label'],

           s=35, edgecolor='b', cmap=plt.cm.Set1)

ax.set_xlabel('Age')

ax.set_ylabel('Annual Income (k$)')

ax.set_zlabel('Spending Score (1-100)')

ax.set_title('3D view of K-Means 5 clusters')

ax.dist = 12



plt.show()
trace3d = go.Scatter3d(x= cust['Age'], y= cust['Spending Score (1-100)'], z= cust['Annual Income (k$)'],

    mode='markers', marker=dict(color = cust['cls_label'], size= 5, opacity=0.7))



layout = go.Layout(

    title = '3D View of Customer Segmentation',

    margin=dict(l=65, r=50, b=65, t=90),

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data = trace3d, layout = layout)

fig.show()