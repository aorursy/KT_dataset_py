#Import Librarires

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# read the dataset

emp_df = pd.read_csv("../input/employee-analysis/employee.csv", sep=",", encoding="ISO-8859-1", header=0)

emp_df.head()
#To display all the columns in DF

pd.set_option('display.max_columns', None)

emp_df.head()
emp_df.info()
#Age column rename 

emp_df['Age']=emp_df['ï»¿Age']
#Age column old name dropping

emp_df.drop('ï»¿Age',axis=1,inplace=True)

emp_df.info()
emp_df.head()
emp_df['EmployeeNumber'].value_counts()
#Attrition,OverTime

# List of variables to map



varlist =  ['Attrition', 'OverTime']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['Attrition'].value_counts()
emp_df['OverTime'].value_counts()
emp_df['JobRole'].value_counts()
#JobRole

# List of variables to map



varlist =  ['JobRole']



# Defining the map function

def binary_map(x):

    return x.map({'Sales Executive': 0, "Research Scientist": 1,'Laboratory Technician':2,'Manufacturing Director':3,'Healthcare Representative':4,'Manager':5,'Sales Representative':6,'Research Director':7,'Human Resources':8})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)
emp_df['JobRole'].value_counts()
#Gender

# List of variables to map



varlist =  ['Gender']



# Defining the map function

def binary_map(x):

    return x.map({'Male': 1, "Female": 0})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['Gender'].value_counts()
#Marital Status

# List of variables to map



varlist =  ['MaritalStatus']



# Defining the map function

def binary_map(x):

    return x.map({'Married': 1, "Single": 0,'Divorced':2})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['MaritalStatus'].value_counts()
#Over18

# List of variables to map



varlist =  ['Over18']



# Defining the map function

def binary_map(x):

    return x.map({'Y': 1, "N": 0})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)
emp_df['Over18'].value_counts()
emp_df['EducationField'].value_counts()
#EducationField

# List of variables to map



varlist =  ['EducationField']



# Defining the map function

def binary_map(x):

    return x.map({'Life Sciences': 1, "Medical": 0,'Marketing':2,'Technical Degree':3,'Other':4,"Human Resources":5})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['EducationField'].value_counts()
emp_df.info()
emp_df['Department'].value_counts()
#Department

# List of variables to map



varlist =  ['Department']



# Defining the map function

def binary_map(x):

    return x.map({'Research & Development': 1, "Sales": 0,'Human Resources':2})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['Department'].value_counts()
emp_df['BusinessTravel'].value_counts()
#BusinessTravel

# List of variables to map



varlist =  ['BusinessTravel']



# Defining the map function

def binary_map(x):

    return x.map({'Travel_Rarely': 1, "Travel_Frequently": 0,'Non-Travel':2})



# Applying the function to the housing list

emp_df[varlist] = emp_df[varlist].apply(binary_map)

emp_df['BusinessTravel'].value_counts()
emp_df.info()
sns.boxplot(emp_df['Age'])
sns.boxplot(emp_df['DailyRate'])
sns.boxplot(emp_df['HourlyRate'])
sns.boxplot(emp_df['MonthlyRate'])
sns.boxplot(emp_df['MonthlyIncome'])
sns.boxplot(emp_df['PercentSalaryHike'])
sns.boxplot(emp_df['TotalWorkingYears'])
sns.boxplot(emp_df['PerformanceRating'])
sns.boxplot(emp_df['NumCompaniesWorked'])
sns.boxplot(emp_df['DistanceFromHome'])
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) # heuristic from article [1]

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H

hopkins(emp_df.drop('EmployeeNumber',axis=1))
emp_df_data=emp_df.drop('EmployeeNumber',axis=1)

# instantiate

scaler = StandardScaler()



# fit_transform

emp_df_scaled = scaler.fit_transform(emp_df_data)

emp_df_scaled.shape
emp_df_scaled = pd.DataFrame(emp_df_scaled)

emp_df_scaled.columns = emp_df_data.columns

emp_df_scaled.head()
emp_df.head()
#PerformanceRating,Percentage Salary hike,Monthly Rate

cluster1=emp_df_scaled[['PerformanceRating','PercentSalaryHike','MonthlyRate']]

cluster1.head()
# elbow-curve/SSD

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(cluster1)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd)
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(cluster1)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(cluster1, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    

    
# final model with k=2

kmeans = KMeans(n_clusters=2, max_iter=50,random_state=50)

kmeans.fit(cluster1)
kmeans.labels_
emp_df['cluster_id'] = kmeans.labels_

emp_df.head()
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

cluster_type = emp_df.groupby(['cluster_id'])['PerformanceRating'].mean().reset_index()

sns.barplot(x = 'cluster_id', y='PerformanceRating', data=cluster_type)

plt.subplot(1,2,2)

cluster_type = emp_df.groupby(['cluster_id'])['PercentSalaryHike'].mean().reset_index()

sns.barplot(x = 'cluster_id', y='PercentSalaryHike', data=cluster_type)

plt.show()
#PerformanceRating,Age,Monthly Rate

cluster2=emp_df_scaled[['PerformanceRating','Age','MonthlyRate']]

cluster2.head()
# elbow-curve/SSD

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(cluster2)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd)
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(cluster2)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(cluster2, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    

    
# final model with k=3

kmeans = KMeans(n_clusters=3, max_iter=50,random_state=50)

kmeans.fit(cluster2)
kmeans.labels_
emp_df['cluster_label'] = kmeans.labels_

emp_df.head()
#plot data with seaborn

sns.scatterplot(x = 'PerformanceRating', y = 'Age', hue = 'cluster_label', data = emp_df, palette = 'Set1')