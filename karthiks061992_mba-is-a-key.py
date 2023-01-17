# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

dataset.columns

#dataset.set_index('sl_no')

#setting sl_no as the index

features=['gender','ssc_p','hsc_p','degree_p','specialisation','mba_p','salary','status']

dataset=dataset[features]

dataset.head(10)



#Iam a begineer looking for some datasets.
#pre-processing data

dataset.isnull().sum()

#only salary has null values 

#67 people are unplaced

dataset['salary']=dataset['salary'].fillna(0).astype(float)

dataset.head()

#kind of got rid of null values 
dataset.groupby(['gender','status'])['status'].count().plot.line()
#from the above analysis we have a decision that says males are placed on a higher ratio compared to females 

#Analysing on specialization

dataset.groupby(['specialisation','gender'])['gender'].count().sort_values(ascending=False)
dataset.groupby(['specialisation','gender'])['gender'].count().sort_values(ascending=False).head(5).plot.line()

#alright so males have higher chances of getting a job in mkt and finance and females have a higher chance in mkt and HR given probably they dont apply for finance
dataset.columns



table=pd.crosstab(dataset.mba_p,dataset.salary)

import scipy.stats

analysis=scipy.stats.chi2_contingency(table)

print(analysis)

#this proves that giving mba scores play a vital role in the salary since there is 66 percent dependency 

#let us analyze this dependency 


dataset.columns
from sklearn.cluster import KMeans

X=dataset.iloc[:,[5,6]].values

wcss=[]

for i in range(1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300)

    kmeans.fit(X)

    kmeans.predict(X)

    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,11),wcss)

#from wcss elbow method we can take 3 clusters into consideration

kmeans=KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=0)

kmeans.fit(X)

y_kmeans=kmeans.predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red')

plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue')

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green')

plt.show()

#This at some point proves that the number of people getting jobs doing mba is higher comparitively to people who have not done mba 

#hence mba plays a key role 
#This at some point proves that the number of people getting jobs doing mba is higher comparitively to people who have not done mba 

#hence mba plays a key role 