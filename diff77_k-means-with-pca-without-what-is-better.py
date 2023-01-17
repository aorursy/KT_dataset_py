import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')
# preview the data
train.head()
train.drop(['Ticket','Fare','Cabin','Name','PassengerId'], axis=1, inplace=True)
train.head()
train.nunique()
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train.isnull().sum()
train.dropna(inplace=True)
train[['Pclass','SibSp','Parch','Embarked']]=train[['Pclass','SibSp','Parch','Embarked']].astype(str) 
train= pd.get_dummies(train)
train.head()
y=train['Survived']
data=train.drop('Survived',axis=1)
data = StandardScaler().fit_transform(data)
kmeans = KMeans(n_clusters=2, random_state=123, algorithm='elkan')

kmeans.fit(data)
clusters = kmeans.predict(data)
cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters
cluster_df['class'] = y
sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')
f1_score(y, clusters)
mPCA = PCA(n_components=20)
PrincipleComponents = mPCA.fit_transform(data)
variance = mPCA.explained_variance_ratio_
variance_ratio = np.cumsum(np.round(variance, decimals=3)*100)
variance_ratio
plt.title("PCA components VS percentage of variance explained")
plt.ylabel("Percentage (%)")
plt.xlabel("# of components")
plt.plot(variance_ratio)
PCAdata = PrincipleComponents[:,:15]

kmeans.fit(PCAdata )
clusters = kmeans.predict(PCAdata )

cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters
cluster_df['class'] = y
sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')
f1_score(y, clusters)
f1scores=[]
for i in  range(2,20):
    PCAdata = PrincipleComponents[:,:i]
    kmeans.fit(PCAdata )
    clusters = kmeans.predict(PCAdata )
    f1score=f1_score(y, clusters)
    f1scores.append(f1score)
    print('PCA dimensions: {}, f1 score {}'.format(i, f1score))
    
plt.plot(range(2,20),f1scores)