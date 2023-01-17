import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding='latin1') 
data.head()
data.drop_duplicates(inplace=True) 
data.drop('date',axis=1,inplace =True) 
data = data.reset_index(drop=True)
plt.figure(figsize=(15,10))

sns.stripplot(x="state", y="number",data=data,

              jitter=True,

              hue="year")#, palette='coolwarm')
plt.figure(figsize=(15,10))

ax=sns.distplot(data.number)

ax = ax.set(yticklabels=[],title='Histogram of number of fires reported')
sns.jointplot(x='number',y='year',data=data)#kind='scatter' - default
pd.DataFrame(data.groupby(data.year).number.median())
pd.DataFrame(data.groupby(data.state).number.std())
x = data.groupby(data.state).number.median()

plt.figure(figsize=(15,10))

ax = sns.barplot(y=x.index.values,x=x.values)

ax = ax.set(xlabel='Median of fires reported',ylabel='States',title='Fires Reported by State')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
X = np.array(data['number'])
X = X.reshape(-1,1)
kmeans.fit(X)
centers = kmeans.cluster_centers_

centers
kmeans.labels_
# sum_square will be key,value pair for the elbow plot!

sum_square = {}



# Let's test for K from 1 to 10, 

# we can use range() function in the for loop here! 

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k).fit(X)

    # .inertia: Computing Sum of Squared Distances 

    # of samples to their closest cluster center.

    sum_square[k] = kmeans.inertia_ 
plt.plot(list(sum_square.keys()),

         list(sum_square.values()),

         

         # Some figure aesthetics

         linestyle='-', # '-' for Continuous line 

         marker='H', # 'H' for Hexagons 

         color='g', # 'g' for green color

         markersize = 8, # size of the masker

         markerfacecolor='b') # 'b' for blue color
f, ax1 = plt.subplots(nrows=1, 

                             ncols=1, 

                             sharey=True,

                             figsize=(10,6))

# For the fitted one, c = kmeans.labels_  

ax1.set_title('K Means (K = 4)')

ax1.scatter(X,kmeans.labels_,

            cmap='rainbow')