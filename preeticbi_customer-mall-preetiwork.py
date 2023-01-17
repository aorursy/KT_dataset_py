# Call Libraries
# For Handling of Warnings
import warnings
# Handling of "Deprecation Warnings"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# For data manipulations
import numpy as np
import pandas as pd
import re

# For plotting
import seaborn as sns; sns.set(style="white", color_codes=True)
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves

# Modeling Library
# For data processing
from sklearn.preprocessing import StandardScaler

# Split dataset
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

# How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# Class to develop kmeans model
from sklearn.cluster import KMeans

# OS related
import os
# Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Go to folder containing data file
os.chdir("/kaggle/input/customer-segmentation-tutorial-in-python")
# Read csv file
cust_df = pd.read_csv("Mall_Customers.csv")
# Set Maximum Columns Option
pd.options.display.max_columns = 100
# Some DataSet related information
cust_df.shape
cust_df.columns
cust_df.head
# Rename Columns 'Annual Income (k$)' & 'Spending Score (1-100)'
cust_df.rename(columns={'Annual Income (k$)':'Annual_Income_k',
                       'Spending Score (1-100)':'Spending_Score_1to100'},inplace=True)
# Drop 'CustomerID' column
cust_df.drop(columns={'CustomerID'},inplace=True)
#have a Look of Modified Column_names & their dtypes
cust_df.columns
cust_df.dtypes
# Transform 'Gender' column
# Replace Male with 0
# Replace Female with 1
cust_df.Gender[cust_df['Gender']=='Male']=0
cust_df.Gender[cust_df['Gender']=='Female']=1
# Add two categorical columns for analysis purpose
# 'Age_Cat': will categorise age in 'Young','Adult','Elder'
# 'Income_Cat' : will categorise Annual_income in 'Low','Medium' ,'High'
cust_df['Age_Cat']= pd.cut(cust_df['Age'],
                         bins=3,
                         labels=['Young','Adult','Elder'])
cust_df['Income_Cat']= pd.cut(cust_df['Annual_Income_k'],
                         bins=3,
                         labels = ['L','M','H'])
# Explore modified data
cust_df.head
# Relate 'Spending_Score' with 'Age'
# Young category customers' maximum spending score is upto 100.
# Adult category customers have avg spending score range between 0 to 90.
# Elder customers spending score is between 0 to 60 only
sns.relplot(x='Age',y='Spending_Score_1to100',col='Age_Cat',data=cust_df)
group1= cust_df.groupby('Age_Cat').mean
sns.distplot(cust_df['Age'])
# Relate Gender with Spending_Score
# Male spending score is between 22 to 70 whereass Female customers' spending score is 30 to 75.

sns.catplot('Gender','Spending_Score_1to100', data = cust_df, kind = 'box') 
# Strong correlation between Age & Annual_Income_k
sns.jointplot(cust_df.Age, cust_df.Annual_Income_k,        kind = 'hex') 
# Strong correlation between Age & Annual_Income_k
sns.jointplot(cust_df.Age, cust_df.Spending_Score_1to100,        kind = 'kde') 
sns.catplot(x='Income_Cat', y='Spending_Score_1to100', kind='bar',hue='Gender',    data = cust_df) 
#Using for loop to plot all at once
columns = ['Age', 'Annual_Income_k', 'Spending_Score_1to100', 'Gender']
fig = plt.figure(figsize = (10,10))
for i in range(len(columns)):
    plt.subplot(2,2,i+1)
    sns.distplot(cust_df[columns[i]])
# More such relationships through for-loop
columns = ['Age', 'Annual_Income_k', 'Spending_Score_1to100', 'Gender']
catVar = ['Age_Cat', 'Income_Cat' ]

# Now for loop. First create pairs of cont and cat variables
mylist = [(cont,cat)  for cont in columns  for cat in catVar]
mylist

# 6.4 Now run-through for-loop
fig = plt.figure(figsize = (10,10))
for i, k in enumerate(mylist):
    #print(i, k[0], k[1])
    plt.subplot(4,2,i+1)
    sns.boxplot(x = k[1], y = k[0], data = cust_df)
# Relationship of a categorical to numeric variable
sns.barplot(x = 'Age_Cat',
            y = 'Annual_Income_k',
            estimator = np.sum,      # As there are multiple occurrences of Gender, sum up 'Clicked_on_ad'
            ci = 95,                 # Estimate default confidence interval using bootstrapping
            hue = 'Gender',
            data = cust_df,
            #capsize = 1
            )

# Relationship of a categorical to another numeric variable
sns.barplot(x = 'Gender',
            y = 'Annual_Income_k',
            estimator = np.sum,      # As there are multiple occurrences of Gender, sum up 'Clicked_on_ad'
            ci = 95,                 # Estimate default confidence interval using bootstrapping
            hue = 'Age_Cat',
            data = cust_df,
            #capsize = 1
            )
# Relationship of a categorical to another categorical variable
fig = plt.figure(figsize = (10,8))
sns.barplot(x = 'Income_Cat',
            y = 'Gender',
            estimator = np.sum,      # As there are multiple occurrences of Gender, sum up 'Clicked_on_ad'
            ci = 68,                 # Estimate default confidence interval using bootstrapping
            hue = 'Age_Cat',
            data = cust_df,
            #capsize = 1
            )
# Drop Categorical Columns
cust_df.drop(columns=['Age_Cat','Income_Cat'], inplace=True)
# Scale data using StandardScaler
ss = StandardScaler()     # Create an instance of class
ss.fit(cust_df)                # Train object on the data
X = ss.transform(cust_df)      # Transform data
X[:5, :]                  # See first 5 rows
# Perform clsutering
gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
# Train the algorithm
gm.fit(X)
# Where are the clsuter centers
gm.means_
# Did algorithm converge?
gm.converged_
# How many iterations did it perform?
gm.n_iter_
# Clusters labels
gm.predict(cust_df)
#  Weights of respective gaussians.
gm.weights_
# What is the frequency of data-points
np.unique(gm.predict(X), return_counts = True)[1]/len(X)
# Plot cluster and cluster centers
fig = plt.figure()
plt.scatter(X[:,0],X[:,1],
            c=gm.predict(X),
            s=2)
plt.scatter(gm.means_[:,0], gm.means_[:,1],
            marker='v',
            s=5,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()

# Discover How many clusters are there?
bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(X)
    bic.append(gm.bic(X))
    aic.append(gm.aic(X))
# Look at the plots

fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8], aic)
plt.plot([1,2,3,4,5,6,7,8], bic)
plt.show()

# Plot has minimum value at 2-clusters
tsne = TSNE(n_components = 2,perplexity=30.0)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='v',
            s=10,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(X)   # Colour as per gmm
            )
# Anomaly detection
densities = gm.score_samples(X)
densities

density_threshold = np.percentile(densities,4)
density_threshold
anomalies = X[densities < density_threshold]
anomalies
anomalies.shape
# Show anomalous points
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='v',
            s=20,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='blue'
            )
plt.show()

# Lets analyse the differences in anomalous & unanomalous (ie normal)data.
# Get first unanomalous data
unanomalies = X[densities >= density_threshold]
unanomalies.shape    
# Transform both anomalous and unanomalous data to pandas DataFrame
df_anomalies = pd.DataFrame(anomalies, columns = cust_df.columns.values)
df_anomalies['type_unA_An'] = 'anomalous'   # Create a IIIrd constant column
df_normal = pd.DataFrame(unanomalies, columns = cust_df.columns.values)
df_normal['type_unA_An'] = 'unanomalous'    # Create a IIIrd constant column

# Explore df_anomalies & df_normal
df_anomalies
df_normal
# Let us see density plots
# Dispersion of Normal points is lesser than anomalous points
sns.distplot(df_anomalies['Annual_Income_k'])
sns.distplot(df_normal['Annual_Income_k'])
# Draw side-by-side boxplots
# Ist stack two dataframes
df = pd.concat([df_anomalies,df_normal])
# Draw featurewise boxplots
sns.boxplot(x = df['type_unA_An'], y = df['Annual_Income_k'])
# Again less dispersion for normal points comparetively .
# Here less dispersion for anomalous points comparetively .
sns.boxplot(x = df['type_unA_An'], y = df['Spending_Score_1to100'])
