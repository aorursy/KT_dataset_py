"""
Last amended: 27th June, 2020
Objectives:

1. Understanding basics of GMM
i)    Read dataset and rename columns appropriately
ii)   Drop customerid column and also transform Gender column to [0,1]
iii)  Use seaborn to understand each feature and relationships among features.
iv)  Use sklearn's StandardScaler() to scale dataset
v)   Perform clustering using Gaussian Mixture Modeling.
vi)  Use aic and bic measures to draw a scree plot and discover ideal number of clusters
viii) Lookup anomalous customers and try to understand their behavior.


"""
# 1.0 Call libraries
#%reset -f                       # Reset memory
# 1.1 Data manipulation library
import pandas as pd
import numpy as np
# 1.2 OS related package
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import seaborn as sns
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

import re

# 1.3 For plotting
import matplotlib.pyplot as plt
import matplotlib
# Install as: conda install -c plotly plotly 
import plotly.express as px


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Import GaussianMixture class
from sklearn.mixture import GaussianMixture

import time

# 1.1 Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )
#1) Read file 'Mall_Customers.csv' 
df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")


# i) Read dataset and rename columns appropriately

cv = df.columns.values # Copy column names to another DF
cv # Display new DF 

#2.1 Clean Column Names by replacing/removing special characters
j = 0
for i in cv:
    cv[j] = re.sub(' ', '_', cv[j]) # Replace space with _
    cv[j] = re.sub('\'','', cv[j])  # Replace apostrophe with blank
    cv[j] = re.sub('[*|\(\)\{\}]','', cv[j]) # Replace special characters
    cv[j] = re.sub('/','_', cv[j])    # Replace / with _
    cv[j] = re.sub('&','_', cv[j])    # Replace & with _
    cv[j] = re.sub('-','_', cv[j])    # Replace - with _    
    cv[j] = re.sub('\.','', cv[j])    # Replace . with _
    cv[j] = re.sub('[,]','', cv[j])   # Replace , with blank         
    cv[j] = re.sub('__.','_', cv[j])  # Replace multiple _ with single _          
    j = j + 1

# Show cleaned column names
cv
# Make a disctionary of Old & new column names
y = dict(zip(df.columns.values, cv))
y

# i) Read dataset and rename columns appropriately
df.rename(
         y,
         inplace = True,
         axis = 1             # Note the axis keyword. By default it is axis = 0
         )

# Show the new column names
df.columns.values
# ii) Drop customerid column and also transform Gender column to [0,1]
# ii) Drop customerid column 
df.head() # Before dropping Customer_ID
cid = df['CustomerID'].values
df.drop(columns = ['CustomerID'], inplace = True)
df.head() # After dropping Customer_ID
df.Gender.unique()
# ii) Drop customerid column and also transform Gender column to [0,1]
# ii) transform Gender column to [0,1]

def trf_gender(x):
    if x == 'Male':
        return 0            # Male = 0
    if x == 'Female':
        return 1            # Female = 1


df["Gender_Transformed"] = df["Gender"].map(lambda x : trf_gender(x))   
# ii) Drop customerid column and also transform Gender column to [0,1]
# Show the transformed DataFrame
df.head()
df.Gender_Transformed.unique()
# iii)  Use seaborn to understand each feature and relationships among features.
# Select numeric column heads
columns = list(df.select_dtypes(include = ['float64', 'int64']).columns.values)
columns
#iii)  Use seaborn to understand each feature and relationships among features.
# 1. Using for loop to plot distribution plots all at once

fig = plt.figure(figsize = (10,10))
for i in range(len(columns)):
    plt.subplot(2,2,i+1)
    sns.distplot(df[columns[i]])
# iii)  Use seaborn to understand each feature and relationships among features.
# 2.0 Relationship of numeric variable with a categorical variable
# 2.1 such relationships through for-loop
columns = ['Age', 'Annual_Income_k$', 'Spending_Score_1_100']
catVar = ['Gender']


# 2.2 Now for loop. First create pairs of cont and cat variables
mylist = [(cont, cat)  for cont in columns  for cat in catVar]
mylist

# 2.3 Now run-through for-loop
fig = plt.figure(figsize = (20,20))
for i, k in enumerate(mylist):
    plt.subplot(4,2,i+1)
    sns.boxplot(x = k[1], y = k[0], data = df)
# iii)  Use seaborn to understand each feature and relationships among features.
# Relationship of numeric to numeric variables
numcolumns = list(df.select_dtypes(include = ['float64', 'int64']).columns.values)

sns.jointplot(df[numcolumns[0]], df[numcolumns[1]], kind = "hex")

sns.jointplot(df[numcolumns[0]], df[numcolumns[2]], kind = "kde")

sns.jointplot(df[numcolumns[0]], df[numcolumns[3]])

sns.jointplot(df[numcolumns[1]], df[numcolumns[2]], kind = "reg")

sns.jointplot(df[numcolumns[1]], df[numcolumns[3]])

sns.jointplot(df[numcolumns[2]], df[numcolumns[3]])


# iii)  Use seaborn to understand each feature and relationships among features.
sns.barplot(x = 'Gender',
            y = 'Spending_Score_1_100',
            estimator = np.mean,
            ci = 95,
            data =df
            )
#iv)  Use sklearn's StandardScaler() to scale dataset
# create a dataframe of numeric columns
nc = df.select_dtypes(include = ['float64', 'int64']).copy()
nc.head()
df.head()
#iv)  Use sklearn's StandardScaler() to scale dataset
# Drop Categorical & discrete columns
df.head() # Before dropping Gender & Gender_Transformed
df.drop(columns = ['Gender'], inplace = True)
gnd = df['Gender_Transformed'].values
df.drop(columns = ['Gender_Transformed'], inplace = True)
df.head() # After dropping Customer_ID
gnd
#iv)  Use sklearn's StandardScaler() to scale dataset

ss = StandardScaler()     # Create an instance of class
ss.fit(df)                # Train object on the data
df.shape
X = ss.transform(df)      # Transform data
X[:5, :]                  # See first 5 rows
X.shape

#iv)  Use sklearn's StandardScaler() to scale dataset
X_train, X_test, _, gnd_test = train_test_split( X,               # np array without target
                                               gnd,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.1 Examine the results
X_train.shape              # (150, 3)
X_test.shape               # (50, 3)

#iv)  Use sklearn's StandardScaler() to scale dataset
clf = KMeans(n_clusters = 3)
# 5.2 Train the object over data
clf.fit(X_train)

# 5.3 So what are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape         # (3, 3)
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   # 150
clf.inertia_                       # Sum of squared distance to respective centriods, SSE 194.72070292819043

silhouette_score(X_train, clf.labels_)    # 0.3742884754041953

#iv)  Use sklearn's StandardScaler() to scale dataset
# Make prediction over our test data and check accuracy
gnd_pred = clf.predict(X_test)
gnd_pred
# How good is prediction
np.sum(gnd_pred == gnd_test)/gnd_test.size # 0.42
#iv)  Use sklearn's StandardScaler() to scale dataset
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = gnd_pred)

#iv)  Use sklearn's StandardScaler() to scale dataset
sse = []
for i,j in enumerate(range(10)):
    # 7.1.1 How many clusters?
    n_clusters = i+1
    # 7.1.2 Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # 7.1.3 Train the kmeans object over data
    clf1.fit(X_train)
    # 7.1.4 Store the value of inertia in sse
    sse.append(clf1.inertia_ )

# 7.2 Plot the line now
sns.lineplot(range(1, 11), sse)
#iv)  Use sklearn's StandardScaler() to scale dataset
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

#iv)  Use sklearn's StandardScaler() to scale dataset
# Intercluster distance: Does not work
from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

#v)   Perform clustering using Gaussian Mixture Modeling.
gm_mall = GaussianMixture(
                           n_components = 3,   # More the clusters, more the time
                           n_init = 10,
                           max_iter = 100
                         )

#v)   Perform clustering using Gaussian Mixture Modeling.
start = time.time()
gm_mall.fit(df)
end = time.time()
(end - start)/60     # 0.0015 minutes

#v)   Perform clustering using Gaussian Mixture Modeling.
# Did algorithm(s) converge?
gm_mall.converged_     # True

#v)   Perform clustering using Gaussian Mixture Modeling.
# Clusters labels
gm_mall.predict(df)
#v)   Perform clustering using Gaussian Mixture Modeling.
# How many iterations did they perform?
gm_mall.n_iter_      #  5
#v)   Perform clustering using Gaussian Mixture Modeling.
#  What is the frequency of data-points
#       for the three clusters. (np.unique()
#       ouputs a tuple with counts at index 1)

np.unique(gm_mall.predict(X), return_counts = True)[1]/len(X)
#v)   Perform clustering using Gaussian Mixture Modeling.
# GMM is a generative model.
#     Generate a sample from each cluster
#     ToDo: Generate digits using MNIST

gm_mall.sample()
#v)   Perform clustering using Gaussian Mixture Modeling.
# Plot cluster and cluster centers
#     both from kmeans and from gmm

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1],
            c=gm_mall.predict(X),
            s=2)

plt.scatter(gm_mall.means_[:, 0], gm_mall.means_[:, 1],
            marker='v',
            s=5,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()

#viii) Lookup anomalous customers and try to understand their behavior.
#     Anomaly detection
#     Anomalous points are those that
#     are in low-density region
#     Or where density is in low-percentile
#     of 4%
#     score_samples() method gives score or
#     density of a point at any location.
#     Higher the value, higher its density

densities = gm_mall.score_samples(X)
densities
#viii) Lookup anomalous customers and try to understand their behavior.
density_threshold = np.percentile(densities,4)
density_threshold
#viii) Lookup anomalous customers and try to understand their behavior.
anomalies = X[densities < density_threshold]
anomalies
anomalies.shape
#viii) Lookup anomalous customers and try to understand their behavior.
# Show anomalous points
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = gm_mall.predict(X))
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='x',
            s=50,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()
#viii) Lookup anomalous customers and try to understand their behavior.
# Get first unanomalous data
unanomalies = X[densities >= density_threshold]
unanomalies.shape    # (192, 3)
#viii) Lookup anomalous customers and try to understand their behavior.
# Transform both anomalous and unanomalous data
#     to pandas DataFrame
df_anomalies = pd.DataFrame(anomalies, columns = ['x', 'y', 'p'])
df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column
df_normal = pd.DataFrame(unanomalies, columns = ['x','y', 'p'])
df_normal['z'] = 'unanomalous'    # Create a IIIrd constant column

#viii) Lookup anomalous customers and try to understand their behavior.
# Let us see density plots
sns.distplot(df_anomalies['x'])
sns.distplot(df_normal['x'])
#viii) Lookup anomalous customers and try to understand their behavior.
# Let us see density plots
sns.distplot(df_anomalies['y'])
sns.distplot(df_normal['y'])
#viii) Lookup anomalous customers and try to understand their behavior.
# Let us see density plots
sns.distplot(df_anomalies['p'])
sns.distplot(df_normal['p'])
#viii) Lookup anomalous customers and try to understand their behavior.
# Draw side-by-side boxplots
# Ist stack two dataframes
df = pd.concat([df_anomalies,df_normal])
# Draw featurewise boxplots
sns.boxplot(x = df['z'], y = df['y'])
sns.boxplot(x = df['z'], y = df['x'])
sns.boxplot(x = df['z'], y = df['p'])
#vi)  Use aic and bic measures to draw a scree plot and discover ideal number of clusters
bic = []
aic = []
for i in range(3):
    gm2 = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm2.fit(X)
    bic.append(gm2.bic(X))
    aic.append(gm2.aic(X))

#vi)  Use aic and bic measures to draw a scree plot and discover ideal number of clusters
fig = plt.figure()
plt.plot([1,2,3], aic)
plt.plot([1,2,3], bic)
plt.show()
# t-stochaistic neighbourhood embedding
#     Even though data is already in 2-dimension,
#     for the sake of completion, 
#     darwing a 2-D t-sne plot and colour
#     points by gmm-cluster labels

tsne = TSNE(n_components = 3, perplexity = 30)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='o',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm2.predict(X)   # Colour as per gmm
            )
plt.title('t-SNE visualization');