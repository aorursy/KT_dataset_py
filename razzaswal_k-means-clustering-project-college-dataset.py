import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/college-data/College_Data.csv',index_col=0)
data.head(2)
data.info()
data.describe()
sns.set()
sns.scatterplot(x='Room.Board', y='Grad.Rate', hue='Private', data=data)
sns.scatterplot(x ='Outstate', y= 'F.Undergrad', hue = 'Private', data=data  )
fcdgrd = sns.FacetGrid(data, col='Private')
fcdgrd  = fcdgrd.map(plt.hist, 'Outstate')
data['Private'] = data['Private'].map({
    'Yes':1,
    'No':0
})
plt.figure(figsize=(10,6))
data[data['Private']==1]['Outstate'].hist(alpha = 0.5, color = 'blue', bins = 30, label = 'Private=1')
data[data['Private']==0]['Outstate'].hist(alpha = 0.5, color = 'red', bins = 30, label = 'Private=0')
plt.legend()
plt.xlabel('Out of State Tuition')
plt.figure(figsize=(10,6))
data[data['Private']==1]['Grad.Rate'].hist(alpha = 0.5, color = 'blue', bins = 30, label = 'Private=1')
data[data['Private']==0]['Grad.Rate'].hist(alpha = 0.5, color = 'red', bins = 30, label = 'Private=0')
plt.legend()
plt.xlabel('Graduate Rate')
data[data['Grad.Rate']>100]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop('Private',axis=1))
kmeans.cluster_centers_
data['Cluster'] = data.apply(lambda row: row.Private, axis = 1)
data.head(2)
kmeans.labels_
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(data['Cluster'],kmeans.labels_)
print(classification_report(data['Cluster'],kmeans.labels_))