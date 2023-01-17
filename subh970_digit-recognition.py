# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data=pd.read_csv("../input/train.csv")

data.head()



# Any results you write to the current directory are saved as output.
image=data.iloc[:,1:]

label=data.iloc[:,:1]

kmeans= KMeans(n_clusters=10)

kmeans.fit(image)

la=kmeans.labels_
di={}

for j in range(10):

    d={}

    for i in range(len(label)):

        if kmeans.labels_[i]==j:

            d[label.label[i]]=d.get(label.label[i],0)+1

    di[j]=next(iter(d))

di
# checking the accuracy

c=0

for i in range(len(label)):

    if di[kmeans.labels_[i]]==label.label[i]:

        c+=1

print("accuracy =",c/len(label)*100,"%")
di={}

for j in range(10):

    d={}

    for i in range(len(label)):

        if kmeans.labels_[i]==j:

            d[label.label[i]]=d.get(label.label[i],0)+1

    di[j]=next(iter(d))

di
image=data.iloc[:,1:]

label=data.iloc[:,:1]

from sklearn.neighbors import KNeighborsClassifier as kn

knn= kn(n_neighbors=10)

image.shape
image=data.iloc[:,1:]

label=data.iloc[:,:1]

from sklearn.neighbors import KNeighborsClassifier as kn

knn= kn(n_neighbors=10)

x_train,x_test,y_train,y_test = train_test_split(image,label,test_size = 0.2,random_state = 100) 

knn.fit(x_train,y_train)



predic=knn.predict(x_test)

# metrics.accuracy_score(y_test,predic)
from sklearn import metrics

metrics.accuracy_score(y_test,predic)
df=pd.read_csv("../input/test.csv")

df.head()
predic=knn.predict(df)