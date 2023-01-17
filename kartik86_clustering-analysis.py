# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/attrition.csv")
df.head()
df.isna().sum()
df.columns
df['Engagement Score']=df['Engagement Score (% Satisfaction)'].apply(lambda x: float(x[:-1])/100)

df['Monthly Income normalized']=df['Monthly Income'].apply(lambda x: float(x/df['Monthly Income'].max()))

data=df[['Engagement Score','Monthly Income normalized']].values
import regex as re

df['Tenure']=df['Tenure'].apply(lambda x: float(re.sub(" ",'',x)))
#Clustering data based on monthly income and Engagment score



# create np array for data points

#plt.figure(figsize=(20,5))

points=data

plt.scatter(data[:,0],data[:,1])

#plt.xlim(-15,15)

#plt.ylim(0,75)

plt.show()



# import KMeans

from sklearn.cluster import KMeans



# create kmeans object

kmeans = KMeans(n_clusters=3)

# fit kmeans object to data

kmeans.fit(data)

# print location of clusters learned by kmeans object

print(kmeans.cluster_centers_)

# save new clusters for chart

y_km = kmeans.fit_predict(points)



plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='blue')

plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='red')

plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='black')

plt.show()
#Clustering data based on tenure and engagment score

data=df[['Engagement Score','Tenure']].values

# create np array for data points

points=data

plt.scatter(data[:,0],data[:,1])

#plt.xlim(-15,15)

#plt.ylim(0,75)

plt.show()



# import KMeans

from sklearn.cluster import KMeans



# create kmeans object

kmeans = KMeans(n_clusters=4)

# fit kmeans object to data

kmeans.fit(data)

# print location of clusters learned by kmeans object

print(kmeans.cluster_centers_)

# save new clusters for chart

y_km = kmeans.fit_predict(points)



plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='blue')

plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='red')

plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='black')

plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='yellow')

plt.show()