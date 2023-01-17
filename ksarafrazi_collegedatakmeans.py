
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

df = pd.read_csv('../input/College.csv')
df.head()
#Data visualization
sns.scatterplot(df['Outstate'],df['F.Undergrad'],hue=df['Private'])
sns.scatterplot(df['Room.Board'],df['Grad.Rate'],hue=df['Private'])
sns.set_style('whitegrid')
sns.distplot(df['Outstate'][df['Private']=='Yes'],kde=False,color='r')
sns.distplot(df['Outstate'][df['Private']=='No'],kde=False)
sns.set_style('whitegrid')
sns.distplot(df['Grad.Rate'][df['Private']=='Yes'],kde=False,color='r')
sns.distplot(df['Grad.Rate'][df['Private']=='No'],kde=False)
#There's an error in the data that needs to be corrected.
df[df['Grad.Rate'] > 100] = 100
sns.set_style('whitegrid')
sns.distplot(df['Grad.Rate'][df['Private']=='Yes'],kde=False,color='r')
sns.distplot(df['Grad.Rate'][df['Private']=='No'],kde=False)
#Creating dummy values for college name
College = pd.get_dummies(df['Unnamed: 0'],drop_first = True)
df_unlabled = pd.concat( [df.drop(['Private','Unnamed: 0'],axis=1) , College] , axis =1)

#K means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_unlabled)
#cluster centers
print(kmeans.cluster_centers_)
#Checking the performance
Lables = df['Private']=='Yes'
print(classification_report(kmeans.labels_,Lables))
