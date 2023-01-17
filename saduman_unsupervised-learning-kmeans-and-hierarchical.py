# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualization

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#We read data

data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

#data includes how many rows and columns

data.shape

print("Our data has {} rows and {} columns".format(data.shape[0],data.shape[1]))

#Features name in data

data.columns 
#diplay first 5 rows

data.head()
data.describe(include='all')
sns.pairplot(data, hue="class", markers=["o", "s"])
corr=data.corr()

fig, ax=plt.subplots(1,1,figsize=(12,8))

sns.heatmap(corr,annot=True, linewidth=.5, ax=ax)
plt.figure(figsize=[5,5])

sns.set(style='darkgrid')

ax = sns.countplot(x='class', data=data, palette='Set2')

data.loc[:,'class'].value_counts()


plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()
# KMeans Clustering

data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)



data["label"] = labels



#We draw our data with clusters.We have 0,1 label

plt.scatter(x = data[data.label == 0].pelvic_radius,y = data[data.label == 0].degree_spondylolisthesis , color = "purple")

plt.scatter(x = data[data.label == 1].pelvic_radius,y = data[data.label == 1].degree_spondylolisthesis , color = "cyan")



plt.show()
# cross tabulation table

df = pd.DataFrame({'labels':labels,"class":data['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
# inertia

inertia_list = np.empty(8)

for i in range(1,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()
cluster_data = data.drop('class',axis = 1)
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(cluster_data)

labels = pipe.predict(cluster_data)





cluster_data['label'] = labels

cluster_data['class'] = data['class']



df = pd.DataFrame({'labels':labels,"class":data['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
cluster_data.head(10)
cluster_data = data.drop('class',axis = 1)



from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(cluster_data,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters = 2,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(cluster_data)



data["label_hc"] = cluster
plt.scatter(x = data[data.label_hc == 0].pelvic_radius,y = data[data.label_hc == 0].degree_spondylolisthesis , color = "pink")

plt.scatter(x = data[data.label_hc == 1].pelvic_radius,y = data[data.label_hc == 1].degree_spondylolisthesis , color = "purple")
#We compare our model in graph

fig = plt.figure(figsize = (15,5))



plt.subplot(1, 3, 1)

plt.scatter(x = data[data["class"] == "Normal"].pelvic_radius , y = data[data["class"] == "Normal"].degree_spondylolisthesis,color = "pink")

plt.scatter(x = data[data["class"] == "Abnormal"].pelvic_radius , y = data[data["class"] == "Abnormal"].degree_spondylolisthesis,color = "yellow")

plt.title("original class")



plt.subplot(1, 3, 2)

plt.scatter(x = data[data.label == 0].pelvic_radius,y = data[data.label == 0].degree_spondylolisthesis , color = "pink")

plt.scatter(x = data[data.label == 1].pelvic_radius,y = data[data.label == 1].degree_spondylolisthesis , color = "yellow")

plt.title("kmeans") 



plt.subplot(1, 3, 3)

plt.scatter(x = data[data.label_hc == 0].pelvic_radius,y = data[data.label_hc == 0].degree_spondylolisthesis , color = "pink")

plt.scatter(x = data[data.label_hc == 1].pelvic_radius,y = data[data.label_hc == 1].degree_spondylolisthesis , color = "yellow")

plt.title("hierarchical")



plt.show()