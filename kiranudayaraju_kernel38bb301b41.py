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
## 1. Call libraries

import numpy as np

import pandas as pd

# 1.1. For displaying graphics

import matplotlib.pyplot as plt



import matplotlib

import matplotlib as mpl     # For creating colormaps

import seaborn as sns

# 1.2 For clustering

from sklearn.cluster import KMeans

# 1.4 OS related

import os
# 2.1 Read csv file

mc = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

mc.shape
mc.info()
mc.head ()
new_col_names = {

                 'Annual Income (k$)' :  'AnnualIncome',

                 'Spending Score (1-100)': 'SpendingScore',

                               }
mc.rename(

         new_col_names,

         inplace = True,

         axis = 1             # Note the axis keyword. By default it is axis = 0

         )
mc.head()
###PLots##

sns.distplot(mc.Age, bins = 20, kde=True)
sns.distplot(mc.AnnualIncome, bins = 20, kde=True)
age_male = mc[mc['Gender']=='Male']['Age']

age_female = mc[mc['Gender']=='Female']['Age']
sns.distplot(age_male, bins = 20, kde=True)
sns.distplot(age_female, bins = 20, kde=True)
inc_male = mc[mc['Gender']=='Male']['AnnualIncome']

inc_female = mc[mc['Gender']=='Female']['AnnualIncome']
sns.jointplot(age_male, inc_male, kind = "kde")
sp_male = mc[mc['Gender']=='Male']['SpendingScore']

sp_female = mc[mc['Gender']=='Female']['SpendingScore']

sns.jointplot(age_female, sp_female, kind = "kde")
sns.pairplot(data=mc, hue='Gender')
plt.scatter(mc["Age"], mc["AnnualIncome"])
plt.scatter(mc["Age"], mc["SpendingScore"])
##Clustering the spending score against age

km = KMeans(n_clusters=2)

km
y_predicted = km.fit_predict(mc[['Age', 'SpendingScore']])

y_predicted
mc['clust'] = y_predicted

mc.head()
mc1 = mc[mc.clust==0]

mc2 = mc[mc.clust==1]
plt.scatter(mc1.Age, mc1['SpendingScore'], color='green')

plt.scatter(mc2.Age, mc2['SpendingScore'], color='red')

plt.xlabel('Age')

plt.ylabel('SpendingScore')
km.cluster_centers_
##Scatter plot with centroid

plt.scatter(mc1.Age, mc1['SpendingScore'], color='green')

plt.scatter(mc2.Age, mc2['SpendingScore'], color='red')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black', marker = '*', label ='centorid')
##Scatter plot with centroid

plt.scatter(mc1.Age, mc1['SpendingScore'], color='green')

plt.scatter(mc2.Age, mc2['SpendingScore'], color='red')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black', marker = '*', label ='centorid')

plt.xlabel('Age')

plt.ylabel('SpendingScore')

plt.legend()
#Find cluster by Elbow Curve Method

krng = range(1,10)

sse = []

for k in krng:

    km = KMeans(n_clusters=k)

    km.fit(mc[['Age', 'SpendingScore']])

    sse.append(km.inertia_)
sse
plt.title('Elbow Curve')

plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(krng, sse)
##Scatter plot with cluster 4 

kmean4 = KMeans(n_clusters=4)

kmean4

y_predicted4 = kmean4.fit_predict(mc[['Age', 'SpendingScore']])

y_predicted4

kmean4.inertia_

kmean4.cluster_centers_

kmean4.labels_

mc['ClustLabel'] = kmean4.labels_

mc.head()
mc['ClustLabel'].nunique()
mc1 = mc[mc.ClustLabel==0]

mc2 = mc[mc.ClustLabel==1]

mc3 = mc[mc.ClustLabel==2]

mc4 = mc[mc.ClustLabel==3]
plt.scatter(mc1.Age, mc1['SpendingScore'], color='green')

plt.scatter(mc2.Age, mc2['SpendingScore'], color='red')

plt.scatter(mc3.Age, mc3['SpendingScore'], color='blue')

plt.scatter(mc4.Age, mc4['SpendingScore'], color='yellow')

plt.scatter(kmean4.cluster_centers_[:,0],kmean4.cluster_centers_[:,1],color = 'black', marker = '*',label='centorid')

plt.xlabel('Age')

plt.ylabel('SpendingScore')

plt.legend()
