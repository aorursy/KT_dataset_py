# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
cust = pd.read_csv("../input/Mall_Customers.csv", index_col="CustomerID")
cust.info() # check for missing data
cust.describe()
# devide data by gender

cust_m = cust[cust["Gender"] == "Male"]

cust_f = cust[cust["Gender"] == "Female"]

# visualization function

def GenderGraph(axis_x, axis_y):

    plt.scatter(cust_m[axis_x], cust_m[axis_y], marker="x", color="blue")

    plt.scatter(cust_f[axis_x], cust_f[axis_y], marker="+", color="red")

    plt.ylabel(axis_y)

    plt.xlabel(axis_x)

    plt.legend(["Male","Female"])

    plt.show()
GenderGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 1
GenderGraph("Age", "Spending Score (1-100)") # Graph 2
GenderGraph("Age", "Annual Income (k$)") # Graph 3
cust.groupby("Gender").agg(["min","max","mean", "std"])
# 1) Judging by the graphs and summary, feature Gender does not carry useful information 

# for further clustering, so we can exclude it from the sample(next, we test this hypothesis).

# 2) On graph 1 we can see 5 distinct areas of thickening of objects.

# At first we can apply KMeans to cluster data into 5 clusters.
from sklearn.cluster import KMeans
cust_ng = cust.drop("Gender", axis = 1)

nc = 5 # current number of clusters

def clust_fit(n_cl):

    cluster = KMeans(n_clusters=n_cl, random_state=0).fit(cust_ng)

    cust_ng["Cluster"] = cluster.labels_

clust_fit(nc)
# visualization clusters

def ClusterGraph(axis_x, axis_y):

    color_d = {0:"red", 1:"green", 2:"yellow", 3:"brown", 4:"black", 5:"pink", 6:"orange"}

    for i in range (nc):

        cust_c = cust_ng[cust_ng["Cluster"] == i]

        plt.scatter(cust_c[axis_x], cust_c[axis_y], marker="x", color=color_d[i])

    plt.ylabel(axis_y)

    plt.xlabel(axis_x)

    plt.show()
ClusterGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 4
ClusterGraph("Age", "Spending Score (1-100)") # Graph 5
ClusterGraph("Age", "Annual Income (k$)") # Graph 6
# 1) On the graph 4 we can observe 5 groups of customers differentiated by income (INC) 

# and spending score (SSC):

# BLACK - high SSC, low INC

# GREEN - high SSC, high INC

# RED - low SSC, low INC

# YELLOW - average SSC, average INC

# BROWN - low SSC, high INC

# Of greatest interest to us is the brown cluster, since with a high level of income, 

# its members have low spending score. Target - BROWN (Cluster 3).

# 2) On graph 5, we can see that high SSC (> 60) members are extremely under 43 years old.

# From this we can conclude that the trading strategy should be aimed at attracting more age

# members.
# select and consider the target group in more detail

cust_t = cust_ng[cust_ng["Cluster"] == 3].drop("Cluster", axis=1)

cust_t.describe()
# Characteristics of the target group:

# -Age from 19 to 59

# -Anual income from 70 to 137

# -Spending score from 1 to 39
# Additional experiments
cust_ng = cust.drop("Gender", axis = 1)

nc = 4

clust_fit(nc)

ClusterGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 7
cust_ng = cust.drop("Gender", axis = 1)

nc = 6

clust_fit(nc)

ClusterGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 8
# Increasing and decreasing the number of clusters by 1 does not change the target cluster.

# Next, we consider clustering for each of the genders separately.
cust_ng = cust_m.drop("Gender", axis = 1)

nc = 5

clust_fit(nc)

ClusterGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 9
cust_ng = cust_f.drop("Gender", axis = 1)

nc = 5

clust_fit(nc)

ClusterGraph("Annual Income (k$)", "Spending Score (1-100)") # Graph 10
# For the male gender, the cluster structure has changed somewhat, for the female 

# gender it has remained the same. At the same time, the target cluster for both 

# cases remained unchanged.