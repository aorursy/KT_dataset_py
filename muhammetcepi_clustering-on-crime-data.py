# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random  

import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/crime-data-from-2010-to-present/Crime_Data_from_2010_to_Present.csv")

df.head()
df.columns
df.drop(columns=["DR Number","Date Reported","Date Occurred","Area Name","Crime Code Description","Weapon Description","Crime Code 1","Crime Code 2","Crime Code 3","Crime Code 4","Address","Cross Street","Premise Description","Weapon Used Code","Status Code","Location ","Status Description","MO Codes"],inplace=True)
df["Victim Sex"].value_counts()
df["Victim Sex"].fillna("M",inplace=True)
df["Victim Descent"].value_counts()
df["Victim Descent"].fillna("H",inplace=True)
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()

le_sex.fit(['M','F','X','H','N','-'])

df["Victim Sex"] = le_sex.transform(df["Victim Sex"].values) 





le_BP = preprocessing.LabelEncoder()

le_BP.fit([ 'H', 'W', 'B','X','A','K','F','C','I','J','P','U','V','Z','G','S','D','L','-','O'])

df["Victim Descent"] = le_BP.transform(df["Victim Descent"].values)





X = df[["Time Occurred","Area ID","Reporting District","Crime Code","Victim Age","Victim Sex","Victim Descent","Premise Code"]].head(1000).values

X[0:5]
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
clusterNum = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
df["Clus_km"] = labels

df.head(5)
df.groupby('Clus_km').mean()
cost =[] 

for i in range(1, 11): 

    KM = KMeans(n_clusters = i, max_iter = 500) 

    KM.fit(X) 

      

    # calculates squared error 

    # for the clustered points 

    cost.append(KM.inertia_)      

  

# plot the cost against K values 

plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 

plt.xlabel("Value of K") 

plt.ylabel("Sqaured Error (Cost)") 

plt.show() # clear the plot 

  

# the point of the elbow is the  

# most optimal value for choosing k 