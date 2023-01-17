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



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
file1 = pd.read_csv ('../input/geospiza-scandens-beak-evolution/finch_beaks_1975.csv')
file2 = pd.read_csv ('../input/geospiza-scandens-beak-evolution/finch_beaks_2012.csv')
file1.head ()
file2.head ()
file1.columns = ['band', 'species', 'blength', 'bdepth']
file1 ['year'] = 1975
file2 ['year'] = 2012
df = pd.concat ([file1, file2])
df.head ()
df.shape
df.info ()
df.isnull ().sum ()
print (round (df.groupby ('species')[['blength', 'bdepth']].describe ().transpose (),2))
df.species.value_counts ()
df_fortis = df[df.species == 'fortis']


fig, ax = plt.subplots (1, 2, figsize=(22, 7), sharex=True)



sns.set_style ('ticks')



plt.subplot (1,2,1)

sns.swarmplot (x = 'species', y = 'blength', data = df_fortis , hue = 'year', dodge = True,)

plt.title ('Beak length comparison for fortis', fontsize = 20)



plt.subplot (1,2,2)

sns.swarmplot (x = 'species', y = 'bdepth', data = df_fortis, hue = 'year', dodge = True)

plt.title ('Beak depth comparison for fortis', fontsize = 20)
df_scandens = df[df.species == 'scandens']
fig, ax = plt.subplots (1, 2, figsize=(22, 7), sharex=True)



sns.set_style ('ticks')



plt.subplot (1,2,1)

sns.swarmplot (x = 'species', y = 'blength', data = df_scandens , hue = 'year', dodge = True,)

plt.title ('Beak length comparison for scandens', fontsize = 20)



plt.subplot (1,2,2)

sns.swarmplot (x = 'species', y = 'bdepth', data = df_scandens, hue = 'year', dodge = True)

plt.title ('Beak depth comparison for scandens', fontsize = 20)
def ecdf (data):

    x = np.sort (data)

    n = len (data)

    y = np.arange (1, n+1)/n 

    return x,y
df_fortis = df_fortis.set_index ('year')
df_scandens = df_scandens.set_index ('year')
x10,y10 = ecdf (df_fortis.loc ['1975','blength'])

x20,y20 = ecdf (df_fortis.loc ['2012','blength'])



x11,y11 = ecdf (df_fortis.loc ['1975','bdepth'])

x21,y21 = ecdf (df_fortis.loc ['2012','bdepth'])





x30,y30 = ecdf (df_scandens.loc ['1975','blength'])

x40,y40 = ecdf (df_scandens.loc ['2012','blength'])



x31,y31 = ecdf (df_scandens.loc ['1975','bdepth'])

x41,y41 = ecdf (df_scandens.loc ['2012','bdepth'])
fig, ax = plt.subplots (3,3, figsize = (15,7))



ax [0,0].hist (df_fortis.loc ['1975','blength'])

ax [0,1].hist (df_fortis.loc ['2012','blength'])

ax [0,2].hist (df_fortis.loc ['1975','bdepth'])

ax [1,0].hist (df_fortis.loc ['2012','bdepth'])

ax [1,1].hist (df_scandens.loc ['1975','blength'])

ax [1,2].hist (df_scandens.loc ['2012','blength'])

ax [2,0].hist (df_scandens.loc ['1975','bdepth'])

ax [2,1].hist (df_scandens.loc ['2012','bdepth'])



plt.tight_layout ()
fig, ax = plt.subplots (1, 2, figsize=(22, 7), sharex=True)

sns.set_style ('ticks')

plt.subplot (1,2,1)

sns.lineplot (x10,y10, data = df_fortis.loc ['1975','blength'])

sns.lineplot (x21,y21, data = df_fortis.loc ['2012','blength'])

plt.title ('Fortis beak length comparison', fontsize = 20)



plt.subplot (1,2,2)

sns.lineplot (x11,y11, data = df_fortis.loc ['1975','bdepth'])

sns.lineplot (x21,y21, data = df_fortis.loc ['2012','bdepth'])

plt.title ('Fortis beak depth comparison', fontsize = 20)
fig, ax = plt.subplots (1, 2, figsize=(22, 7), sharex=True)

sns.set_style ('ticks')

plt.subplot (1,2,1)

sns.lineplot (x30,y30, data = df_scandens.loc ['1975','blength'])

sns.lineplot (x40,y40, data = df_scandens.loc ['2012','blength'])

plt.title ('Scandens beak length comparison', fontsize = 20)



plt.subplot (1,2,2)

sns.lineplot (x31,y31, data = df_scandens.loc ['1975','bdepth'])

sns.lineplot (x41,y41, data = df_scandens.loc ['2012','bdepth'])

plt.title ('Scandens beak depth comparison', fontsize = 20)
# function for generating replicates of means for a given data.

def BS_replicates_mean (data, n_replicates):

    replicates = np.empty (n_replicates, dtype = object)

    for i in range (n_replicates):

        replicates [i] = np.mean (np.random.choice (data, len (data)))

    return replicates
#Mean difference basis observed data

mean_diff_fortis_l = np.mean (df_fortis.loc ['2012','blength']) - np.mean (df_fortis.loc ['1975','blength']) 

mean_diff_fortis_d = np.mean (df_fortis.loc ['2012','bdepth']) - np.mean (df_fortis.loc ['1975','bdepth'])

mean_diff_scandens_l = np.mean (df_scandens.loc ['2012','blength']) - np.mean (df_scandens.loc ['1975','blength']) 

mean_diff_scandens_d = np.mean (df_scandens.loc ['2012','bdepth']) - np.mean (df_scandens.loc ['1975','bdepth'])
#Mean difference replicates

mean_diff_fortis_l_replicates = BS_replicates_mean (df_fortis.loc ['2012','blength'],1000)-BS_replicates_mean (df_fortis.loc ['1975','blength'],1000)

mean_diff_fortis_d_replicates = BS_replicates_mean (df_fortis.loc ['2012','bdepth'],1000)-BS_replicates_mean (df_fortis.loc ['1975','bdepth'],1000)



mean_diff_scandens_l_replicates = BS_replicates_mean (df_scandens.loc ['2012','blength'],1000)-BS_replicates_mean (df_scandens.loc ['1975','blength'],1000)

mean_diff_scandens_d_replicates = BS_replicates_mean (df_scandens.loc ['2012','bdepth'],1000)-BS_replicates_mean (df_scandens.loc ['1975','bdepth'],1000)
#Confidence Intervals

print ('The confidence interval {} for fortis length mean difference {}'.format ( np.percentile (mean_diff_fortis_l_replicates, [2.5,97.5]), mean_diff_fortis_l))

print ('The confidence interval {} for fortis depth mean difference {}'.format ( np.percentile (mean_diff_fortis_d_replicates, [2.5,97.5]), mean_diff_fortis_d))

print ('The confidence interval {} for scandens length mean difference {}'.format ( np.percentile (mean_diff_scandens_l_replicates, [2.5,97.5]), mean_diff_scandens_l))

print ('The confidence interval {} for scandens depth mean difference {}'.format ( np.percentile (mean_diff_scandens_d_replicates, [2.5,97.5]), mean_diff_scandens_d))
#Let's first check this metric for the observed data.

mean_beak_ratio_fortis_1975 = np.mean (df_fortis.loc ['1975','blength']/df_fortis.loc ['1975','bdepth'])

mean_beak_ratio_fortis_2012 = np.mean (df_fortis.loc ['2012','blength']/df_fortis.loc ['2012','bdepth'])



mean_beak_ratio_scandens_1975 = np.mean (df_scandens.loc ['1975','blength']/df_scandens.loc ['1975','bdepth'])

mean_beak_ratio_scandens_2012 = np.mean (df_scandens.loc ['2012','blength']/df_scandens.loc ['2012','bdepth'])

print ('mean_beak_ratio_fortis_1975 : {}'.format (round (mean_beak_ratio_fortis_1975,2)))

print ('mean_beak_ratio_fortis_2012 : {}'.format (round (mean_beak_ratio_fortis_2012,2)))

print ('\n\n')

print ('mean_beak_ratio_scandens_1975 : {}'.format (round (mean_beak_ratio_scandens_1975,2)))

print ('mean_beak_ratio_scandens_2012 : {}'.format (round (mean_beak_ratio_scandens_2012,2)))
fig, axes = plt.subplots (2,2, figsize = (10, 10))



axes [0,0].scatter (x = df_fortis.loc ['1975','blength'], y = df_fortis.loc ['1975','bdepth'])

axes [0,0].set_title ('fortis 1975')



axes [0,1].scatter (x = df_fortis.loc ['2012','blength'], y = df_fortis.loc ['2012','bdepth'] )

axes [0,1].set_title ('fortis 2012')

axes [1,0].scatter (x = df_scandens.loc ['1975','blength'], y = df_scandens.loc ['1975','bdepth'] )

axes [1,0].set_title ('scandens 1975')

axes [1,1].scatter (x = df_scandens.loc ['2012','blength'], y = df_scandens.loc ['2012','bdepth'] )

axes [1,1].set_title ('scandens 2012')

dict = pd.DataFrame ({'fortis 1975':round (np.corrcoef (df_fortis.loc ['1975','blength'],df_fortis.loc ['1975','bdepth'])[0,1],2),'fortis 2012': round (np.corrcoef (df_fortis.loc ['2012','blength'],df_fortis.loc ['2012','bdepth'])[0,1],2), 'scandens 1975': round (np.corrcoef (df_scandens.loc ['1975','blength'],df_scandens.loc ['1975','bdepth'])[0,1],2), 'scandens 2012': round (np.corrcoef (df_scandens.loc ['2012','blength'],df_scandens.loc ['2012','bdepth'])[0,1],2)}, index = [' '])

print (dict)
beak_ratio_fortis_1975 = df_fortis.loc ['1975','blength']/df_fortis.loc ['1975','bdepth']

beak_ratio_fortis_2012 = df_fortis.loc ['2012','blength']/df_fortis.loc ['2012','bdepth']



beak_ratio_scandens_1975 = df_scandens.loc ['1975','blength']/df_scandens.loc ['1975','bdepth']

beak_ratio_scandens_2012 = df_scandens.loc ['2012','blength']/df_scandens.loc ['2012','bdepth']

beak_ratio_fortis_combinedMean = np.mean (np.concatenate ((beak_ratio_fortis_1975.values,beak_ratio_fortis_2012.values)))



beak_ratio_scandens_combinedMean = np.mean (np.concatenate ((beak_ratio_scandens_1975.values,beak_ratio_scandens_2012.values)))
beak_ratio_fortis_1975_shifted = beak_ratio_fortis_1975 - np.mean (beak_ratio_fortis_1975 ) + beak_ratio_fortis_combinedMean

beak_ratio_fortis_2012_shifted = beak_ratio_fortis_2012 - np.mean (beak_ratio_fortis_2012) + beak_ratio_fortis_combinedMean



beak_ratio_scandens_1975_shifted = beak_ratio_scandens_1975 - np.mean (beak_ratio_scandens_1975) + beak_ratio_scandens_combinedMean

beak_ratio_scandens_2012_shifted = beak_ratio_scandens_2012 - np.mean (beak_ratio_scandens_2012) + beak_ratio_scandens_combinedMean
np.mean (beak_ratio_scandens_2012_shifted)
beak_ratio_fortis_1975_replicates = BS_replicates_mean (beak_ratio_fortis_1975_shifted, 10000)

beak_ratio_fortis_2012_replicates = BS_replicates_mean (beak_ratio_fortis_2012_shifted, 10000)



beak_ratio_scandens_1975_replicates = BS_replicates_mean (beak_ratio_scandens_1975_shifted, 10000)

beak_ratio_scandens_2012_replicates = BS_replicates_mean (beak_ratio_scandens_2012_shifted, 10000)


print ('P-value fortis :',np.sum ((beak_ratio_fortis_2012_replicates - beak_ratio_fortis_1975_replicates) >= abs((np.mean (beak_ratio_fortis_2012)- np.mean(beak_ratio_fortis_1975))))/10000)
print ('P-value scandens :',np.sum ((beak_ratio_scandens_2012_replicates - beak_ratio_scandens_1975_replicates) >= abs((np.mean (beak_ratio_scandens_2012)- np.mean(beak_ratio_scandens_1975))))/10000)
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
y = df.species
X = df.drop (['species','band','year'], axis = 1)
X.shape
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3)
K_Means = KMeans (n_clusters=2,

    init='k-means++', n_init = 12)
K_Means.fit (X_train)
k_pred = K_Means.predict (X_test)
k_means_labels = K_Means.labels_ 
k_means_cluster_centers = K_Means.cluster_centers_
pd.crosstab (k_pred, y_test)
# Initialize the plot with the specified dimensions.

fig = plt.figure(figsize=(15, 8))



# Colors uses a color map, which will produce an array of colors based on

# the number of labels there are. We use set(k_means_labels) to get the

# unique labels.

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))



# Create a plot

ax = fig.add_subplot(1, 1, 1)



# For loop that plots the data points and centroids.

# k will range from 0-3, which will match the possible clusters that each

# data point is in.

for k, col in zip(range(2), colors):



    # Create a list of all data points, where the data poitns that are 

    # in the cluster (ex. cluster 0) are labeled as true, else they are

    # labeled as false.

    my_members = (k_means_labels == k)

    

    # Define the centroid, or cluster center.

    cluster_center = k_means_cluster_centers[k]

    

    # Plots the datapoints with color col.

    ax.plot(X_train[my_members]['blength'], X_train[my_members]['bdepth'], 'w', markerfacecolor=col, marker='o')

    

    # Plots the centroids with specified color, but with a darker outline

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=12)



# Title of the plot

ax.set_title('KMeans', fontsize = 30)



# Remove x-axis ticks

ax.set_xticks(())

ax.set_xlabel ('length', fontsize = 15)

# Remove y-axis ticks

ax.set_yticks(())

ax.set_ylabel ('depth', fontsize = 15)

plt.legend ()

# Show the plot

plt.show()
