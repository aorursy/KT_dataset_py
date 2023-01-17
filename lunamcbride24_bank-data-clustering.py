# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans #Clustering use case
from sklearn import metrics as mt #Check metrics on the model
from sklearn.preprocessing import RobustScaler, normalize #Scaling options
from sklearn.decomposition import PCA #Explaining variance

from yellowbrick.cluster import SilhouetteVisualizer #Visualize Silhouette Scores

import matplotlib.pylab as plt #Plotting graphs
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D #Plotting graphs in 3D

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credit = pd.read_csv("../input/ccdata/CC GENERAL.csv") #Load the dataset into a dataframe
credit.head() #Take a peek at the dataframe
print(credit.count()) #Gets the counts of the columns

#There are some with inconsistent values, being CREDIT_LIMIT and MINIMUM_PAYMENTS

print(credit["CREDIT_LIMIT"].isnull().any()) #Check for nulls in CREDIT_LIMIT
print(credit["MINIMUM_PAYMENTS"].isnull().any()) #Check for nulls in MINIMUM_PAYMENTS
credit["MINIMUM_PAYMENTS"] = credit["MINIMUM_PAYMENTS"].fillna(0) #Fills nulls in MINIMUM_PAYMENTS with 0
credit["CREDIT_LIMIT"] = credit["CREDIT_LIMIT"].fillna(credit["CREDIT_LIMIT"].mean()) #Fills the null in CREDIT_LIMIT with the mean limit
print(credit.count()) #Print the counts
metric = ["BALANCE", "PURCHASES", "PAYMENTS", "CUST_ID"] #Laying out the metrics stated above plus cust_id for indexing
metricsDf = credit[metric] #Take the metrics from the credit dataframe
metricsDf = metricsDf.set_index("CUST_ID") #Set the customer ID to the index
metricsDf = metricsDf.astype("float") #Converts the types to float to make kmeans more applicable
metricsDf.head() #Display the metrics dataframe
fig = plt.figure() #Build the figure
ax = Axes3D(fig) #Make it 3D

#Define the x y z to be the dataset columns
x = list(metricsDf.iloc[:,0])
y = list(metricsDf.iloc[:,1])
z = list(metricsDf.iloc[:,2])

#Define the axes labels
names = metricsDf.columns
ax.set_xlabel(names[0])
ax.set_ylabel(names[1])
ax.set_zlabel(names[2])

ax.scatter(x, y, z, c = "purple", marker = "o") #Create a scatterplot of the data
plt.show() #Show the graph
#Scaling with a robust scaler, as messing with it and the min/max and standard scalers showed it to be the best here
robust = RobustScaler()
trainRobust = robust.fit_transform(metricsDf)

fig = plt.figure() #Build the figure
ax = Axes3D(fig) #Make it 3D

#Define the x y z to be the dataset columns
x = trainRobust[:,0]
y = trainRobust[:,1]
z = trainRobust[:,2]

#Define the axes labels
names = metricsDf.columns
ax.set_xlabel(names[0])
ax.set_ylabel(names[1])
ax.set_zlabel(names[2])

ax.scatter(x, y, z, c = "purple", marker = "o") #Create a scatterplot of the robust data
plt.show() #Show the graph
normalRobust = normalize(trainRobust) #Normalize the robust scaled data
fig = plt.figure() #Build the figure
ax = Axes3D(fig) #Make it 3D

#Define the x y z to be the dataset columns
x = normalRobust[:,0]
y = normalRobust[:,1]
z = normalRobust[:,2]

#Define the axes labels
names = metricsDf.columns
ax.set_xlabel(names[0])
ax.set_ylabel(names[1])
ax.set_zlabel(names[2])

ax.scatter(x, y, z, c = "purple", marker = "o") #Create a scatterplot with the normalized data
plt.show() #Show the graph
pcaTrain = PCA().fit(trainRobust) #A PCA for just the fit data
pcaNormal = PCA().fit(normalRobust) #A PCA for the normalized data

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,4)) #Creates a subplot for both explained variances

#Sets the x and y labels on each graph
axes[0].set_xlabel("Component Number")
axes[0].set_ylabel("Percent Variance")
axes[1].set_xlabel("Component Number")
axes[1].set_ylabel("Percent Variance")

#Sets the titles of the graph, one normal, the other non-normal
axes[0].set_title("Non-normal Explained Variance")
axes[1].set_title("Normal Explained Variance")

#Sets a gray grid below for easier reading
axes[0].grid(alpha=0.25)
axes[1].grid(alpha=0.25)
axes[0].set_axisbelow(True)
axes[1].set_axisbelow(True)

#Plots the explained variance for the normal and non-normal data
axes[0].plot(np.cumsum(pcaTrain.explained_variance_ratio_))
axes[1].plot(np.cumsum(pcaNormal.explained_variance_ratio_))

plt.show() #Shows the plots
pcaSet = PCA(n_components = 2) #Initialize a PCA with two components
pcaSet = pcaSet.fit_transform(normalRobust) #Insert the normalized data, squashing it into two components
pcaDf = pd.DataFrame(data = pcaSet, columns = ["Component 1", "Component 2"]) #Insert the PCA set into a dataframe
pcaDf.head() #Take a peek at the dataframe
plt.figure() #Build a figure

#Set the axes to the components
axes[0].set_xlabel("Component 1")
axes[0].set_ylabel("Component 2")
axes[0].set_title("2 Component PCA")

#Take a look at the orb
plt.scatter(pcaDf["Component 1"], pcaDf["Component 2"], c="purple")
results = {}
numClusters = 11

for k in range(2 , numClusters):
    print("-"*100) #Separate the iterations
    results[k] = {} #Collect the results at K
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(pcaDf) #Run the kmeans for k
    silhouette = mt.silhouette_score(pcaDf, kmeans.labels_, metric = "euclidean") #Get the silhouette score
    
    #Put the scores into the results dictionary
    results[k]["Silhouette"] = silhouette
    results[k]["Score"] = kmeans.score
    results[k]["Inertia"] = kmeans.inertia_
    
    #Print the results
    print("Clusters: {}".format(k))
    print("Silhouette Score: {}".format(silhouette))
clusters = [2, 3, 4, 5, 6, 7, 8] #Choose a couple clusters to visualize
for cluster in clusters:
    kmeans = KMeans(n_clusters = cluster, random_state = 0) #Run KMeans
    visualize = SilhouetteVisualizer(kmeans) #Set a visualizer for KMeans
    visualize.fit(pcaDf) #Fit the visualizer to the data
    visualize.poof() #Display the visualization
plt.figure() #Build a figure

#Set the axes to the components
axes[0].set_xlabel("Component 1")
axes[0].set_ylabel("Component 2")
axes[0].set_title("2 Component PCA")

#Take a look at the orb
plt.scatter(pcaDf["Component 1"], pcaDf["Component 2"], c = KMeans(n_clusters = 3).fit_predict(pcaDf), cmap =plt.cm.winter) 
plt.show() 
fig = plt.figure() #Build the figure
pcaSetTrain = PCA(n_components = 2) #Initialize a PCA with two components
pcaSetTrain = pcaSetTrain.fit_transform(metricsDf) #Insert the normalized data, squashing it into two components
pcaDfTrain = pd.DataFrame(data = pcaSetTrain, columns = ["Component 1", "Component 2"]) #Insert the PCA set into a dataframe

#Scatter based off the original set
plt.scatter(pcaDfTrain["Component 1"], pcaDfTrain["Component 2"], c = KMeans(n_clusters = 3).fit_predict(pcaDfTrain), cmap =plt.cm.winter, marker = "o") 
plt.show() 
info = metricsDf.copy() #Get a copy of the original data
info["Cluster"] = KMeans(n_clusters = 3).fit_predict(pcaDf) #Add a column for which cluster each was classified as
info.head() #Show the new dataframe
sums = [0, 0, 0, 0, 0, 0, 0, 0, 0] #A list to hold the sum of each classification, 0-2 for cluster 0, 3-5 for cluster 1, and 6-8 for 2
counts = [0, 0, 0] #Counts for each cluster, the cluster number being the same as the index.
clust = info["Cluster"].copy() #Get the cluster classifications

#For loop to get the sums of each field based off each cluster classification
for i in range(0,len(clust)):
    currentCluster = clust[i] #Get the current cluster
    
    #If in cluster 0
    if currentCluster == 0:
        #Sum the balances, purchases, and payments for cluster 0
        sums[0] = sums[0] + info["BALANCE"][i]
        sums[1] = sums[1] + info["PURCHASES"][i]
        sums[2] = sums[2] + info["PAYMENTS"][i]
        counts[0] += 1
        
    #If in cluster 1
    elif currentCluster == 1:
        #Sum the balances, purchases, and payments for cluster 1
        sums[3] = sums[3] + info["BALANCE"][i]
        sums[4] = sums[4] + info["PURCHASES"][i]
        sums[5] = sums[5] + info["PAYMENTS"][i]
        counts[1] += 1
    
    #If in cluster 2
    else:
        #Sum the balances, purchases, and payments for cluster 2
        sums[6] = sums[6] + info["BALANCE"][i]
        sums[7] = sums[7] + info["PURCHASES"][i]
        sums[8] = sums[8] + info["PAYMENTS"][i]
        counts[2] += 1
    
mean0 = [sums[0]/counts[0], sums[1]/counts[0], sums[2]/counts[0]] #Calculate the means for cluster 0
mean1 = [sums[3]/counts[1], sums[4]/counts[1], sums[5]/counts[1]] #Calculate the means for cluster 1
mean2 = [sums[6]/counts[2], sums[7]/counts[2], sums[8]/counts[2]] #Calculate the means for cluster 2

#Print the mean arrays for each cluster
print("             Balance             Purchases         Payments")
print("Cluster 0: ", mean0)
print("Cluster 1: ", mean1)
print("Cluster 2: ", mean2)