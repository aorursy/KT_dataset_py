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

        

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook")

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_wine

from sklearn.datasets import load_boston



from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from yellowbrick.cluster import KElbowVisualizer

from yellowbrick.cluster import SilhouetteVisualizer

from yellowbrick.cluster import InterclusterDistance

from yellowbrick.model_selection import LearningCurve



eu=pd.read_excel('/kaggle/input/euindicators/EUIndicators-2014-2018.xlsx',index_col=0)  #Import the dataset and set the Country column as the index





#Check data

eu.head()



#Check Datatype of the columns 



eu.dtypes #All float values



#Checking for null values



null = eu[eu.isna().any(axis=1)]

null #Ireland has 5 NaN values 





#Replace the Service Confidence Indicator NaN value of Luxembourg with the mean value of the column

eu[' Service Confidence Indicator '].fillna((eu[' Service Confidence Indicator '].mean()), inplace=True)



#Dropping Ireland row



eu.dropna(inplace=True)



eu_scaled_fit=MinMaxScaler().fit(eu)

eu_scaled=eu_scaled_fit.transform(eu)



#The scaled array

eu_scaled[:5]



#Converting the array to a Dataframe



EU_scaled=pd.DataFrame(eu_scaled)

EU_scaled.head()





plt.figure(figsize=(12,9))



# Now we apply KMeans

model = KMeans(random_state=39)



# we want first to find out how many clusters using the elbow technique

visualizer = KElbowVisualizer(model, k=(1,8),timings=False)

visualizer.fit(eu_scaled)       

visualizer.show()



#The elbow plot shows the number of clusters to be 4 

# Now we use Silhoutte to visualize the compactness of the clusters



plt.figure(figsize=(12,9))



model = KMeans(4, random_state=39)

model.fit(EU_scaled)



visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(EU_scaled)      

visualizer.show()
# Inter cluster distance 



plt.figure(figsize=(12,9))



visualizer = InterclusterDistance(model, min_size=10000, random_state=39, legend_loc='upper left')



visualizer.fit(EU_scaled)

visualizer.show()
 # import hierarchical clustering libraries

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



plt.figure(figsize=(17,9))



# create dendrogram

dn = sch.dendrogram(sch.linkage(eu_scaled, method='ward'), labels=eu.index)

plt.show()



#Printing the 4 clusters

print('Cluster 1: Czech Republic, Malta, Sweden')

print('Cluster 2: Netherlands, Denmark,Bulgaria, Romania, Estonia, Slovakia,. Hungary, UK, Croatia, Slovenia')

print('Cluster 3: Luxembourg, Austria, Belgium, Germany, Finland')

print('Cluster 4: Cyprus, Portugal, Spain, Lithuania, Italy, Latvia, Greece, France, Poland')
cc=pd.read_csv('/kaggle/input/credit-cards/CCData.csv',index_col=0)



#Filling NaN values in the Minimum Payments column with 0

cc['MINIMUM_PAYMENTS'].fillna(0,inplace=True)



cc.dropna(inplace=True)

cc_scaled_fit=MinMaxScaler().fit(cc)

cc_scaled=cc_scaled_fit.transform(cc)



#The scaled array

cc_scaled[:5]



# Converting the array to a Dataframe



CC_scaled=pd.DataFrame(cc_scaled)

CC_scaled.head()





plt.figure(figsize=(12,9))



# Now we apply KMeans

model = KMeans(random_state=39)



# we want first to find out how many clusters using the elbow technique

visualizer = KElbowVisualizer(model, k=(1,8))

visualizer.fit(eu_scaled)       

visualizer.show()
# Now we use Silhoutte to visualize the compactness of the clusters



plt.figure(figsize=(12,9))



model = KMeans(4, random_state=39)

model.fit(CC_scaled)







visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(CC_scaled)      

visualizer.show()
# Inter cluster distance 



plt.figure(figsize=(12,9))



#model=MiniBatchKMeans(n_clusters=3).fit(X_scaled)



visualizer = InterclusterDistance(model, min_size=10000, random_state=39, legend_loc='upper left')

#visualizer = InterclusterDistance(model)

visualizer.fit(CC_scaled)

visualizer.show()
 # import hierarchical clustering libraries

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



plt.figure(figsize=(17,9))



# create dendrogram

dn = sch.dendrogram(sch.linkage(cc_scaled, method='ward'), labels=cc.index)

plt.show()


