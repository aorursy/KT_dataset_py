from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
def species_label(theta):

    if theta == 0:

        return raw_data.target_names[0]

    if theta == 1:

        return raw_data.target_names[1]

    if theta == 2:

        return raw_data.target_names[2]

raw_data = datasets.load_iris()

data_desc = raw_data.DESCR 

# print(data_desc)

'''DESCR function is used to show description of the dataset'''

data = pd.DataFrame(raw_data.data, columns = raw_data.feature_names)

print(data)

data['species'] = [species_label(theta) for theta in raw_data.target]

data['species_id'] = raw_data.target

data.head()
data.tail()
data.describe()
data.pivot_table(index='species', values=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','species_id'],aggfunc=np.mean)
d_corr = data.iloc[:,[0, 1, 2, 3, 4, 5]].corr()

d_corr
sns.set_style('whitegrid')

sns.pairplot(data, hue='species')

plt.show()
X_data = data.drop(['species', 'species_id'], axis = 1) 
kmeans = KMeans(n_clusters=6)

y = kmeans.fit_predict(X_data)

print(y)
kmeans.inertia_
clus_num=[]

for clusters in range(1, 10):

    kmeans = KMeans(n_clusters = clusters).fit(X_data)

    kmeans.fit(X_data)

    clus_num.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 10), clus_num)

plt.title('Elbow method')

plt.xlabel('Cluster Number')

plt.ylabel('Inertia')

plt.show()
def cluster_1_label(alpha):

    if alpha == 0:

        return 'setosa'

    if alpha == 1:

        return 'virginica'

    if alpha == 2:

        return 'versicolor'

kmeans_model_1 = KMeans(n_clusters=3,random_state=123)

distances_1 = kmeans_model_1.fit_transform(data.iloc[:,0:4])

labels_1 = kmeans_model_1.labels_

data['cluster_1']=labels_1

data['cluster_1_label']=data['cluster_1'].apply(cluster_1_label)

with sns.color_palette("hls", 8):

    sns.pairplot(data.iloc[:,[0,1,2,3,-2]], hue='cluster_1')
pd.crosstab(data['species'],labels_1)
cluster_1_accuracy = len(data[data['species']==data['cluster_1_label']])/len(data)

print('K=3 KMeans -> {0:.4f}%'.format(cluster_1_accuracy*100))