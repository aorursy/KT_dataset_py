#Thanks to Ashwini Sarode for her detailed tutorial on Principle Component Analysis

#https://github.com/AshwiniRS/Medium_Notebooks/blob/master/PCA/PCA_Iris_DataSet.ipynb





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

from sklearn.cluster import KMeans

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#Importing data but dropping the unlabeled first column 

data = pd.read_csv('../input/data.csv', index_col=0)
#Finding NaNs in the dataset

data.isna().sum()
#Displaying data columns

print(data.columns)
#Displaying shape and discription of the data

print(data.shape)

print(data.describe())
#Total number of players from each country

pl.figure(figsize =(20,50))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Nationality']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=14)
#Most comman age players

pl.figure(figsize =(25,15))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Age']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=14)
#Players with common heights

pl.figure(figsize =(30,15))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Height']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=18)
#Players with common weights

pl.figure(figsize =(30,25))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Weight']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=18)
#Players with common playing position

pl.figure(figsize =(30,10))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Position']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=18)
#Players with common overall ratings

pl.figure(figsize =(30,20))

pl.ylabel('ylabel', fontsize=20)

pl.xlabel('Total count of players', fontsize=20)

plot = data.groupby(['Overall']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=18)
#Players with common peferred foot

pl.figure(figsize =(10,5))

pl.ylabel('ylabel', fontsize=12)

pl.xlabel('Total count of players', fontsize=12)

plot = data.groupby(['Preferred Foot']).ID.count().plot('barh')

plot.tick_params(axis='both', which='major', labelsize=12)
#Dropping unrequired columns

data.drop(['ID', 'Name','Age', 'Photo','Nationality', 'Flag', 'Overall','Potential','Club','Club Logo', 'Value', 'Wage','Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type','Real Face', 'Position', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'Release Clause'], axis=1, inplace=True)

#Dropping rows with missing data

data.dropna(axis=0, how='any', inplace=True)
#Correlation matrix & Heatmap

pl.figure(figsize =(20,20))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
#Dropping uncorrelated columns

data.drop(columns=['GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes','LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB'], inplace=True)

data.columns
#Calculating the mean value of the whole featureset

cols = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle'];

mean = 0

for i in range(0,len(cols)):

    mean += data[cols[i]].mean()

aggr_mean = round((mean/2900)*100, 2);

print(aggr_mean)
#Labeling data on the basis of whole featureset mean value; If player's total score is > aggr_mean then Above-average Players Else 'Below-average Players'

labels=pd.DataFrame(np.where(round((data.sum(axis=1)/2900)*100,2) > aggr_mean, 'Above-average Players', 'Below-average Players'))
#Mean normalization of data

data = data.sub(data.mean(axis=0), axis=1)
#Converting dataframe to matrix

data_mat = np.asmatrix(data)

data_mat
# sigma = 1/df_a_mat.shape[0] * np.dot(df_a_mat.transpose(),df_a_mat)

sigma = np.cov(data_mat.T)

sigma
sigma.shape
#Calculating eigen values and eigen vectors

eigVals, eigVec = np.linalg.eig(sigma)
#Sorting eigen values in decreasing order

sorted_index = eigVals.argsort()[::-1] 

eigVals = eigVals[sorted_index]

eigVec = eigVec[:,sorted_index]

eigVals
eigVec
#To reduce dimensions of the data set from 29 features to 2 features, we select the top 2 eigen vectors

eigVec = eigVec[:,:2]

eigVec
#Transforming data into new samplespace

eigVec = pd.DataFrame(np.real(eigVec))

transformed = data_mat.dot(eigVec)

transformed
#Combining the transformed data with its respective labels

final_data = np.hstack((transformed, labels))

final_data = pd.DataFrame(final_data)

final_data.columns = ['pc1', 'pc2', 'label']
#Plotting the transformed data

groups = final_data.groupby('label')

figure, axes = plt.subplots()

axes.margins(0.05)

for name, group in groups:

    axes.plot(group.pc1, group.pc2, marker='o', linestyle='', ms=6, label=name)

    axes.set_title("PCA on fifa19 dataset")

axes.legend()

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()
#Formatting data for K-means cluster analysis

dataK = np.array(list(zip(final_data['pc1'], final_data['pc2'])))

dataK
# Initializing KMeans

kmeans = KMeans(n_clusters=2)

# Fitting with input

kmeans = kmeans.fit(dataK)

# Predicting the clusters

labels = kmeans.predict(dataK)

# Getting the cluster centers

C = kmeans.cluster_centers_

C
#Plotting the clusters

figure, axes = plt.subplots()

axes.margins(0.05)

axes.scatter(dataK[:, 0], dataK[:, 1], c=labels)

axes.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=100)

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.show()
#Transforming the labels to numerical values for computing accuracy, classification report and confusion matrix

final_data['label']=np.where(final_data['label']=='Above-average Players', 1, 0)

labels_array=np.array(final_data['label'])

labels_array
#Computing the K-means clusters Accuracy

print("K-means Accuracy:",metrics.accuracy_score(labels_array, labels))

#Computing the error.

print("Mean Absoulte Error:", mean_absolute_error(labels, labels_array))

#Computing classification Report

print("Classification Report:\n", classification_report(labels_array, labels))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(labels_array, labels),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)