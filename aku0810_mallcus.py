import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df = pd.read_csv('../input/mall-customers/Mall_Customers.csv')

df.head()
df.describe()
#Let's do Data Exploration Analaysis 

sns.countplot(x='Genre', data=df)

plt.title('Distribution of Gender Equality')
#About Age

df.hist('Age',bins=35)

plt.title('Distribution of Age')

plt.xlabel('Age')
plt.hist('Age', data=df[df['Genre']=='Male'],alpha=0.5, label='Male')

plt.hist('Age', data=df[df['Genre']=='Female'],alpha=0.5,label='Female')

plt.title('Distribution of Ages By Gender')

plt.xlabel('Age')

plt.legend()
df.hist('Annual Income (k$)')

plt.title('Annual Income Distribution in Thousands of Dollars')

plt.xlabel('Thousands of Dollars')
plt.hist('Annual Income (k$)',data=df[df['Genre']=='Male'],alpha=0.5, label='Male')

plt.hist('Annual Income (k$)',data=df[df['Genre']=='Female'],alpha=0.5,label='Female')

plt.title('Distribution of Income by Gender')

plt.xlabel('Income(Thousands of Dollars)')

plt.legend()
#Comparing Spending Score

male_cutomers = df[df['Genre']=='Male']

female_customers = df[df['Genre']=='Female']



print(male_cutomers['Spending Score (1-100)'].mean())

print(female_customers['Spending Score (1-100)'].mean())
sns.scatterplot('Age','Annual Income (k$)', hue='Genre',data=df)

plt.title('Age to Income, Colored by Gender')
sns.heatmap(df.corr(),annot=True)
sns.heatmap(male_cutomers.corr(),annot=True)

plt.title('Correlation Heatmap = Male')
sns.heatmap(female_customers .corr(),annot=True)

plt.title('Correlation Heatmap = Female')
sns.lmplot('Age', 'Spending Score (1-100)',data=female_customers )

plt.title('Age to Spending Score, Female Only' )

          

   

sns.lmplot('Age', 'Spending Score (1-100)',data=male_cutomers  )

plt.title('Age to Spending Score, Male Only' )
dataset = pd.read_csv('../input/mall-customers/Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X , method = 'ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters= 5, affinity = 'euclidean', linkage= 'ward') #n_cluster is set 5 as it is optimal as shown in dendogram

y_hc= hc.fit_predict(X)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
