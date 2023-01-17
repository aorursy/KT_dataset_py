import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/college-data/College_Data',index_col=0)
df.head()
df.info()
df.describe()
sns.scatterplot(x='Grad.Rate',data=df,y='Room.Board',hue='Private')
sns.scatterplot(x='F.Undergrad',data=df,y='Outstate',hue='Private')
sns.FacetGrid(df,hue="Private",palette='coolwarm',aspect=2).map(plt.hist,'Outstate',bins=20,alpha=0.7)
sns.FacetGrid(df,hue="Private",palette='coolwarm',aspect=2).map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]
sns.FacetGrid(df,hue="Private",palette='coolwarm',aspect=2).map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(df.drop('Private',axis=1))
model.cluster_centers_
df['Cluster'] = df['Private'].apply(lambda x: 1 if x == "Yes" else 0)
df['Cluster']
df.head()
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],model.labels_))
print(classification_report(df['Cluster'],model.labels_))