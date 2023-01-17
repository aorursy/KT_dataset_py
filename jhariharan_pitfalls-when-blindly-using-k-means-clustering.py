# importing python libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning tools

from sklearn.cluster import KMeans



import os



# read the dataset

df = pd.read_csv('../input/bikeshare.csv')
# randomly sample 5 rows from data to get an idea for how it is formatted

df.sample(5)
cmap = plt.get_cmap('tab20')

months = ['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']

df['Month_Num'] = np.zeros(len(df))

fig = plt.figure(figsize=(15,10))

fig.suptitle('Monthly Temperature Distributions')

for i in range(0,12):

    plt.subplot(4,3,i+1)

    plt.hist(df[df['MONTH']==months[i]]['TEMP'],histtype='stepfilled',bins=20,color=cmap(i),range=[0,1])

    plt.title(months[i])

    df['Month_Num'].loc[df['MONTH']==months[i]] = i+1   # assigning column of month numbers



plt.tight_layout(pad=4.0)
temp_array = df['TEMP'].values     # pull the temperature data out as a vector

temp_array = temp_array.reshape(-1,1)     # reshape the vector into an array

kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(temp_array)     # apply the K-Means clustering

df['Temp_Clusters'] = kmeans     # store the kmeans cluster values in the DataFrame
plt.figure(figsize=(15,5))

plt.scatter(df['MONTH'],df['TEMP'],c=df['Temp_Clusters'])

plt.title('Clustered Temperature Data Plotted by Month')

plt.ylabel('Normalized Temperature')

plt.xlabel('Month')

plt.show()
temp_array = np.array([df['TEMP'].values, df['Month_Num']]).transpose()   # pull the temperature and month numbers into a numpy array

kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(temp_array)     # apply the K-Means clustering

df['Temp_Clusters'] = kmeans     # store the kmeans cluster values in the DataFrame
plt.figure(figsize=(15,5))

plt.scatter(df['MONTH'],df['TEMP'],c=df['Temp_Clusters'])

plt.title('Clustered Temperature Data Plotted by Month')

plt.ylabel('Normalized Temperature')

plt.xlabel('Month')

plt.show()
temp_array = df['Month_Num'].values     # pull the month data out as a vector

temp_array = temp_array.reshape(-1,1)     # reshape the vector into an array

kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(temp_array)     # apply the K-Means clustering

df['Temp_Clusters'] = kmeans     # store the kmeans cluster values in the DataFrame

plt.figure(figsize=(15,5))

plt.scatter(df['MONTH'],df['TEMP'],c=df['Temp_Clusters'])

plt.title('Clustered Temperature Data Plotted by Month')

plt.ylabel('Normalized Temperature')

plt.xlabel('Month')

plt.show()