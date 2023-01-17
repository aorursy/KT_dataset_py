# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import matplotlib.pyplot as plt # Matplotlib for Plotting
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#Load the dataset
dataset=pd.read_csv('../input/Bank-Customers.csv')
dataset.head(n=10) # Reading the first 10 rows of the dataset

dataset.info() #Checking the type of data
dataset.isnull().sum() #Checking null values.As there is no missing data, hence we can process the dataset
#Dropping the Cusd_id column,as this is unnecessary in the dataset
dataset.drop('Cust_id',axis=1,inplace=True) #axis=1 means removing column wise
#inplace=True means after dropping col, inorder to reflect changes in dataframe we are using inplace
dataset.head(n=10) #Again checking first 10 rows of the dataset
#Selecting Earning and Credit Score column as i want to do prediction on those columns and assigning to variable X
X=dataset.iloc[:,[1,2]].values
#Now using the dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.axhline(y=200, color='#851e3e', linestyle='--')
fig=plt.gcf()
fig.set_size_inches(13,9)
sns.set_style('darkgrid')
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
#Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering #importing the required library
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#Visualizing the clusters using scatter plots
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],marker='x')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],marker='*')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],marker='P')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],marker='H')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],marker='+')
plt.title('Bank Customers_Earning VS Credit Score')
fig=plt.gcf()
fig.set_size_inches(13,9)
sns.set_style('darkgrid')
plt.xlabel('Earning')
plt.ylabel('Credit Score')
plt.show()
