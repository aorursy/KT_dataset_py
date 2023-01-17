# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/groceries/Groceries.csv")

data.tail()
unique = pd.DataFrame(data["Item"].unique(),columns=["Itemsets"])

print("Number of unique itemsets is",len(data["Item"].unique()))
lis = []

for i in unique.Itemsets:

    lis.append(data[data.Item == i].count()[1])

    
unique["Count"] = lis

unique.head()
len(data["Customer"].unique())
item_list = data.groupby("Customer").Item.apply(list).values.tolist()

x = []

for i in item_list:

    x.append(len(i))

plt.hist(x,range = (0,32))

plt.title("Histogram of number of unique items")

plt.xlabel("Number of items in basket")

plt.ylabel("Count")

plt.savefig("Histogram.png")
df_ = pd.DataFrame(x)

df_.describe()
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

from mlxtend.preprocessing import TransactionEncoder

# Find the frequent itemsets

trans_en = TransactionEncoder()

te_ary = trans_en.fit(item_list).transform(item_list)

ItemIndicator = pd.DataFrame(te_ary, columns=trans_en.columns_)



# Calculate the frequency table

frequent_itemsets = apriori(ItemIndicator, min_support = 75/9835, use_colnames = True)
frequent_itemsets
y = []

for i in range(0,len(frequent_itemsets["itemsets"])):

    y.append(len(frequent_itemsets["itemsets"][i]))

print("Largest k value among the itemset is",max(y))
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

print("The total number of association rules are",len(assoc_rules),"with minimum confidence of 1%")
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])



plt.xlabel("Confidence")

plt.ylabel("Support")

plt.show()

assoc_rules_60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)

assoc_rules_60
df = pd.read_csv("/kaggle/input/spiral/Spiral.csv")

df.head()
sns.scatterplot(df.x,df.y)

plt.title("ScatterPlot for X vs Y")
from sklearn.cluster import KMeans

training_data = df[['x','y']]

kmeans = KMeans(n_clusters=2,random_state=60616).fit(training_data)



centroids = kmeans.cluster_centers_

df["Labels"] = kmeans.labels_
sns.scatterplot(df.x,df.y,hue=df.Labels)

plt.title("Scatterplot specifying 2 clusters")
import math

import sklearn.neighbors



# Three nearest neighbors

kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')

nbrs = kNNSpec.fit(training_data)

d3, i3 = nbrs.kneighbors(training_data)
# Retrieve the distances among the observations

distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')

distances = distObject.pairwise(training_data)
# Create the Adjacency and the Degree matrices

nObs = df.shape[0]

Adjacency = np.zeros((nObs, nObs))

Degree = np.zeros((nObs, nObs))
for i in range(nObs):

    for j in i3[i]:

        if (i <= j):

            Adjacency[i,j] = math.exp(- distances[i][j])

            Adjacency[j,i] = Adjacency[i,j]



for i in range(nObs):

    sum = 0

    for j in range(nObs):

        sum += Adjacency[i,j]

    Degree[i,i] = sum

        

Lmatrix = Degree - Adjacency



from numpy import linalg as LA

evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters

plt.scatter(np.arange(0,9,1), evals[0:9,])

plt.xlabel('Sequence')

plt.ylabel('Eigen value')

plt.title("Eigen Value Plot")

plt.show()
Z = evecs[:,[0,1]]



plt.scatter(Z[[0]], Z[[1]])

plt.xlabel('X - Axis')

plt.ylabel('Y - Axis')

plt.title("Eigen Vectors Plot")

plt.show()
#print("The means and standard deviation of the eigenvectors are as follows :","\n\nMeans\t\t   ",round(Z[[0]].mean(),10) , round(Z[[1]].mean(),10) ,"\n\nStandard Deviations", round(Z[[0]].std(),10) , round(Z[[1]].std(),10))

pd.DataFrame([[Z[[0]],0.0707106781,0.0707106781],[Z[[1]],-0.0707106781,0.0707106781]],columns=["EigenVectors","Mean","Standard Deviation"])
kmeans_spectral = KMeans(n_clusters=2, random_state=0).fit(Z)



df['Labels'] = kmeans_spectral.labels_



#plt.scatter(df['x'], df['y'], c = df['SpectralCluster'])

sns.scatterplot(df['x'], df['y'],hue = df['Labels'])

plt.xlabel('X - axis')

plt.ylabel('Y - axis')

plt.title("Spectral Clusters")

plt.legend(loc=0)

plt.show()