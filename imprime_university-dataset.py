import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/College_Data.csv",index_col=0)
df.head()
df.info()
df.describe()
sns.pairplot(df)
sns.set_style('whitegrid')

sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',palette='magma',size=10,aspect=1,fit_reg=False,)
sns.set_style('whitegrid')

sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',

           palette='magma',size=10,aspect=1,fit_reg=False)
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='viridis',size=6,aspect=2)

g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='viridis',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]
sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, max_iter=500)
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_
def converter(cluster):

    if cluster=='Yes':

        return 1

    else:

        return 0
df['Cluster'] = df['Private'].apply(converter)
df.head()
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(df['Cluster'],kmeans.labels_))

print(classification_report(df['Cluster'],kmeans.labels_))