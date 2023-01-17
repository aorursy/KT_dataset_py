import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

file = '../input/College.csv'

df = pd.read_csv(file)
df.head()
df.info()
sns.set_style('whitegrid')

sns.lmplot(x = 'Room.Board',y = 'Grad.Rate',data=df, hue='Private',

           palette='coolwarm',height=5,aspect=1,fit_reg=False)
sns.set_style('whitegrid')

sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',

           palette='coolwarm',height=6,aspect=1,fit_reg=False)
sns.set_style('whitegrid')

sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',

           palette='coolwarm',height=6,aspect=1,fit_reg=False)
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2)

col = df.drop('Unnamed: 0',axis = 1)

kmeans.fit(col.drop('Private',axis=1))
kmeans.cluster_centers_
kmeans.labels_
def converter(cluster):

    if cluster=='Yes':

        return 1

    else:

        return 0
col['Cluster'] = col['Private'].apply(converter)
col.head()
col['Cluster'].shape
kmeans.labels_.shape
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(col['Cluster'],kmeans.labels_))

print(classification_report(col['Cluster'],kmeans.labels_))