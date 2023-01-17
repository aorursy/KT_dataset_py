# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image

url="https://i.stack.imgur.com/FQhxk.jpg"

Image(url,width=800, height=800)
url="https://i.stack.imgur.com/vc01j.png"

Image(url,width=800, height=800)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import cufflinks as cf

cf.go_offline()

import seaborn as sns
from sklearn.datasets import make_blobs

#This is the we will use to create artificial dataset
data=make_blobs(n_samples=200, n_features=2,centers=4, cluster_std=1.8,random_state=101)

data
data[0].shape # The data contains 200 rows and 2 columns
plt.figure(figsize=(15,10))

plt.scatter(data[0][:,0], data[0][:,1]) # Here we visualize both of the columns

#this data represents the two blobs we have assigned 
data[1] #This data represents the four clusters we have created with the centers parameter
plt.figure(figsize=(15,10))

plt.scatter(data[0][:,0], data[0][:,1],c=data[1],cmap="rainbow")
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4) #Here I assign 4 because I know that it s 4 in my artificial dataset 

kmeans.fit(data[0]) #The algorithm will fit the features of the data
kmeans.cluster_centers_

#This atrribute returns the centers of the four clusters
kmeans.labels_

#This returns the labels that algorithm find as True
fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(15,10))

ax1.set_title("K Means")

ax1.scatter(data[0][:,0], data[0][:,1],c=kmeans.labels_,cmap="rainbow")

#Here we color the scatter plot according to the kmeans' labels



ax2.set_title("Original")

ax2.scatter(data[0][:,0], data[0][:,1],c=data[1],cmap="rainbow")

#Here we color the scatter plot according to the original labels