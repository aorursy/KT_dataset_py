#Project
#Clustering Universities into two groups
import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/us-college-data/College_Data.csv',index_col=0)
df.head()
df.info()
df.describe()
sns.lmplot(x='Room.Board',y='Grad.Rate', data=df,hue='Private')
#Don't want fit lines
sns.lmplot(x='Room.Board',y='Grad.Rate', data=df,hue='Private',fit_reg=False,palette='coolwarm',size=6,aspect=1)
#Cost/Graduation rate for Private/Public Colleges
#Lmplot - linear model plot
sns.lmplot(x='Outstate',y='F.Undergrad', data=df,hue='Private',size=6,aspect=1)
#Tuition higher for private schools. Public higher student body.
#Stacked Histogram
g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Outstate',bins=20,alpha=.7)
#Cost of tuition for out of state students and comparing private vs. non-private colleges
#alpha makes more see through. Cost of private.
g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=.7)
#Graduation rate? How over 100%?
df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College'] = 100
#Warning - A value is trying to be set on a copy of a slice from a dataframe
#Warning says affecting original data frame.
df[df['Grad.Rate']>100]
#OK now.
g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=.7)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
#Use KMeans and make 2 clusters from the data
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_

#Cluster center vectors
#apply method off of dataframe
def converter(private):

    if private =='Yes':

        return 1

    else:

        return 0
df['Cluster'] = df['Private'].apply(converter)
df.head()
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))
#Not bad cause using just features to decide groups.
#Recap
#imported, data, check data, exploratory data analysis
#Scatterplot of outstate/undergrad/costs
#Stacked histogram, addressed error, fixed error, warning
#Checked, kmeans clustering, number of clusters, fit