import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy.stats

import pylab

import seaborn as sns
data_path="../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv"
data=pd.read_csv(data_path)
data.head()
gender_dataframe=pd.get_dummies(data.Gender)
gender_dataframe.head(2)
data.drop(["Gender"],axis=1,inplace=True)
data.insert(loc=2,column="Male",value=gender_dataframe.Male)
data.head()
data.isnull().sum()##checking the null value
data.describe()
plt.hist(data.Age,color="#41b098")

plt.title("Age Distribution throughout the dataset")

data["Age"].plot.kde()

plt.show()
sns.countplot(data.Age)

plt.rcParams['figure.figsize'] = (20, 10)

plt.title("Distribution of the Age")

plt.show()
scipy.stats.probplot(data.Age,plot=pylab)

pylab.show()
from pandas.plotting import andrews_curves
plt.hist(data["Annual Income (k$)"],color="#41b098")

plt.title("Anual Income Distribution throughout the dataset")

data["Annual Income (k$)"].plot.kde()

plt.show()
sns.countplot(data["Annual Income (k$)"])

plt.rcParams['figure.figsize'] = (20, 7)

plt.title("Distribution of the Salary")

plt.show()
scipy.stats.probplot(data["Annual Income (k$)"],plot=pylab)

pylab.show()
#identifying the number of the Males and Females

males=data.Male.sum()

females=200-males

#creating Pie Chart for the 

color=["#80ff00", "#4245a8"]

plt.pie([males,females],labels=["Males","Females"],explode = (0, 0.1,),shadow=True,startangle=360,radius=1.5,colors=color,autopct='%1.1f%%')

fig=plt.gcf()

plt.show()
sns.pairplot(data)

plt.show()
data.head()
features=data.iloc[:,1:]
features.head()
from sklearn.cluster import KMeans
intertia=[]

index_number=[]



for i in range(1,15):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

  kmeans.fit(features)

  intertia.append(kmeans.inertia_)

  index_number.append(i)
plt.plot(intertia)

plt.show()
from sklearn.metrics import silhouette_score
sil_score=[]

for n_clusters in range(2,12):

    clusterer = KMeans (n_clusters=n_clusters).fit(features)

    preds = clusterer.predict(features)

    centers = clusterer.cluster_centers_



    score = silhouette_score (features, preds, metric='euclidean')

    sil_score.append(score)

    
plt.plot(sil_score)

plt.title("Silhouette score vs No.Of cluste")

plt.show()
kmeans=KMeans(n_clusters=4,max_iter=300)

kmeans.fit(features)
class_per_data=kmeans.labels_
print("Classes are",class_per_data)