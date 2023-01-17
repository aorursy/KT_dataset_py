import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'
data= pd.read_csv('../input/mtcars.csv')
data
data.shape
data.info()
data.describe()
data.dtypes
data.isna().any()
data.isna().sum()

#No missing values
#univariate Analysis

data.hist(grid=False, figsize=(20,10), color='pink')
#boxplot

for a in data:

    if (a=='model' or a=='vs' or a=='am'):

        continue

    else:

        plt.figure()

        data.boxplot(column=[a], grid=False)

        
data.head()
#count plot for vs

data['vs'].value_counts()

sns.countplot(data['vs'])
#count plot for vs

data['am'].value_counts()

sns.countplot(data['am'])
#count plot for vs

data['gear'].value_counts()

sns.countplot(data['gear'])
#count plot for vs

data['carb'].value_counts()

sns.countplot(data['carb'])
#count plot for cyl

data['cyl'].value_counts()

sns.countplot(data['cyl'])
#Bivariate analysis

data.corr()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), square=True, linewidths=0.2)

plt.xticks(rotation=90)

plt.yticks(rotation=0)
plt.figure(figsize=(20,10))

sns.pairplot(data, diag_kind='kde')
#let try to use Label Encoder first

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

data['Class']= le.fit_transform(data['model'])
data.head()
X1= data.iloc[:,1:12]

Y1= data.iloc[:,-1]
#lets try to plot Decision tree to find the feature importance

from sklearn.tree import DecisionTreeClassifier

tree= DecisionTreeClassifier(criterion='entropy', random_state=1)

tree.fit(X1, Y1)
imp= pd.DataFrame(index=X1.columns, data=tree.feature_importances_, columns=['Imp'] )

imp.sort_values(by='Imp', ascending=False)
sns.barplot(x=imp.index.tolist(), y=imp.values.ravel(), palette='coolwarm')



#taking only two variable #disp and #qsec as these variable has high importance
X=data[['disp','qsec']]

Y= data.iloc[:,0]
#lets try to create segments using K means clustering

from sklearn.cluster import KMeans

#using elbow method to find no of clusters

wcss=[]

for i in range(1,7):

    kmeans= KMeans(n_clusters=i, init='k-means++', random_state=1)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,7), wcss, linestyle='--', marker='o', label='WCSS value')

plt.title('WCSS value- Elbow method')

plt.xlabel('no of clusters- K value')

plt.ylabel('Wcss value')

plt.legend()

plt.show()
#Here we got no of clusters = 2 

kmeans= KMeans(n_clusters=2, random_state=1)

kmeans.fit(X)
kmeans.predict(X)
#Cluster Center

kmeans.cluster_centers_
data['cluster']=kmeans.predict(X)

data.sort_values(by='cluster').head()
#plotting Cluster plot



plt.scatter(data.loc[data['cluster']==0]['disp'], data.loc[data['cluster']==0]['qsec'], c='green', label='cluster1-0')

plt.scatter(data.loc[data['cluster']==1]['disp'], data.loc[data['cluster']==1]['qsec'], c='red', label='cluster2-1')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='center')

plt.xlabel('disp')

plt.ylabel('qsec')

plt.legend()

plt.show()