# Team Name: 221B Baker Street
# Team Members: Jai Agarwal, Shreyas Nitin Pujari, Guruprasad M
# Members' USNs: 01FB16ECS144, 01FB16ECS371, 01FB16ECS126
# Data Anamytics - Assignment 6

#Technique 1: Clustering (comparison of Agglomerative and K-Means Clustering)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing the dataset
Data = pd.read_csv('../input/Absenteeism_at_work.csv')
#Printing the first 5 lines of the data set
print(Data.head())
#Prints the total number of null values in each column of the dataset
Data.isnull().sum()
#Importing Counter
from collections import Counter
Abs = Data['Absenteeism time in hours']
#print(type(Data['Absenteeism time in hours']))
Seq = Counter(Abs)
#print (Seq)
Keys = list(Seq.keys())
Freq = list(Seq.values())
print ("Classes = ",Keys)  #Each class value corresponds to number of absent hours
print ('Number of classes: ',len(Keys))  #Displays number of distinct classes of hours of absenteeism.
print ("Freq = ",Freq)  #Frequency of each class (Indices correspond, that is, first index of Freq is for First index for Classes and so on)
#print(sorted (Freq))
#Vectors are of 14-Dimension (14-attributes present)

nrows = Data.shape[0]
ncols = Data.shape[1]
#print (nrows, ncols)
#Data_train, Data_test = train_test_split(Data, test_size = 0.3,random_state=42) #
#The below code converts the values of data frame (except absenteeism column) into a 2-D list (List within a list)

DF_List = []
for i in range(nrows):
    Temp = []
    for j in range(ncols-1):
        k = Data.iat[i,j]
        Temp.append(k)
    DF_List.append (Temp)
#Printing the data frame in the form of a list
#print (DF_List)
        
import sklearn.cluster  #For agglomerative clustering
import sklearn.metrics  #For RMSE
from math import *  #For Square root
from scipy import spatial
#Parameters for agglomerative clustering:
#n_clusters = Number of clusters. Here, considered as Keys (=19) because number of classes = 19 (shown above)
#affinity represents the distance metric used. here, Three different metrics have been used to compare them
#Linkage represents the distance to use when a pair of clusters are merged to a single cluster. Here, the average distance is used
#That is, the centroid of the merged cluster is the centroid of the two clusters which got merged into one.
agglomerative1 = sklearn.cluster.AgglomerativeClustering(n_clusters=len(Keys), affinity = 'cosine', linkage = 'average')
agglomerative2 = sklearn.cluster.AgglomerativeClustering(n_clusters=len(Keys), affinity = 'manhattan', linkage = 'average')
agglomerative3 = sklearn.cluster.AgglomerativeClustering(n_clusters=len(Keys), affinity = 'euclidean', linkage = 'average')
Agg1 = agglomerative1.fit_predict(DF_List)
Agg2 = agglomerative2.fit_predict(DF_List)
Agg3 = agglomerative3.fit_predict(DF_List)
#print(Y)
Agg_Res1 = Counter(Agg1)
Agg_Freq1 = list(Agg_Res1.values())
Agg_Freq1 = sorted (Agg_Freq1)
Agg_Res2 = Counter(Agg2)
Agg_Freq2 = list(Agg_Res2.values())
Agg_Freq2 = sorted (Agg_Freq2)
Agg_Res3 = Counter(Agg3)
Agg_Freq3 = list(Agg_Res3.values())
Agg_Freq3 = sorted (Agg_Freq3)
#print(Counter(Agg))
print ('Actual class frequencies\n',sorted(Freq))  #Prints the actual counts of each classes (given)
print ('Frequencies using Cosine distance\n',sorted(Agg_Freq1)) #Prints the counts of each cluster, where each cluster corresponds to a class
print ('Frequencies using Manhattan distance\n',sorted(Agg_Freq2))
print ('Frequencies using Euclidean distance\n',sorted(Agg_Freq3))
#Above 3 lines, sorting has been performed because, the classes generated by agglomerative clustering do not correspond to the actual classes (with respect to list indices)
#Hence, by sorting, we get them to correspond (with reference to list indices)
#print ('RMSE for Cosine distance',sqrt(sklearn.metrics.mean_squared_error(Freq, Agg_Freq1)))
#print ('RMSE for Manhattan distance',sqrt(sklearn.metrics.mean_squared_error(Freq, Agg_Freq2)))
#print ('RMSE for Euclidean distance',sqrt(sklearn.metrics.mean_squared_error(Freq, Agg_Freq3)))

#Cosine similarity  (higher the value, more similar are the two lists)
print ('Cosine similarity for cosine distance',spatial.distance.cosine(Freq, Agg_Freq1))
print ('Cosine similarity for manhattan distance',spatial.distance.cosine(Freq, Agg_Freq2))
print ('Cosine similarity for euclidean distance',spatial.distance.cosine(Freq, Agg_Freq3))
#Comparing with K-Means
#Here, K is considered as 19 as there are 19 classes
kmeans = sklearn.cluster.KMeans(n_clusters=19, random_state=0, n_init = 200, max_iter = 500)
kmeans.fit(DF_List)
clust_labels = kmeans.predict(DF_List)
KMRes = Counter(clust_labels)
KM_freq = list(KMRes.values())
KM_freq = sorted (KM_freq)
print ('Class Frequencies using K-Means\n',KM_freq)
print ('Cosine similarity using K-Means algorithm',spatial.distance.cosine(Freq, KM_freq))

#Results and Analysis

#Therefore, from the above cosine similarities, we can conclude that Agglomerative clustering is better for the given data set
#compared to K-Means Algorithm
#This may be because, the data might not be in a spherical pattern, hence Agglomerative has given better similarity than K-Means

#Also , in Agglomerative clustering, Manhattan distance seems to be the best distance metric among Manhattan, Euclidean and Cosine distances
#(as observed from the Cosine similarity values)

#Using these results, if a new entry is considered, we record the initial frequency value list
#We then add the new entry to the list. And re-run the Agglomerative Clustering Algorithm using Manhattan distance
#We now obtain the new frequency values
#By comparing with old class value frequency list, we can predict the class to which the new value belongs
#Since each class is nothing but a certain number of hours of absenteeism, the value of Absenteeism time in hours
#can be predicted.
#Technique2: Classfication

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Importing dataset
Data = pd.read_csv("../input/Absenteeism_at_work.csv")
#Print the beginning few rows of the dataset
Data.head()
#Describe and summarise basic features of the dataset
Data.describe()
#Data PreProcessing
#Check if any null or missing values
Data.isnull().sum()
#Preparing the training and testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(Data, test_size = 0.3,random_state=42)
x_train = train.drop('Absenteeism time in hours',axis=1)
y_train = train['Absenteeism time in hours']

x_test = test.drop('Absenteeism time in hours', axis=1)
y_test = test['Absenteeism time in hours']
#Scaling all the features in the training dataset to have uniform values..

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)
#from sklearn import preprocessing
#x_scaled_train = preprocessing.scale(x_train)
#x_scaled_test = preprocessing.scale(x_test)
#x_scaled_train.shape
#import required packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
#Running knn for k values from 1 to 100, plotting errors and finding best value for K to fit model
rmse_val = [] #to store rmse values for different k
for K in range(30):
    K = K+1
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(x_train, y_train)  #fit the model
    y_pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
#From curve we can conclude that for k=8 is the best kNN classifier model
#k = 34 is min RMSE value with 14.424

#For the plot, Y axis corresponds to RMSE and X axis corresponds to value of K
#Choosing K=8 as optimal nearest neighbors value
from sklearn import metrics
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("======================================================================\n")
print(classification_report(y_test, y_pred))
y_pred
#Results and Analysis

#For model 2 we used KNN classifier which chooses K nearest neighbors(in this case 8).
#The model works by predicting value of a new datapoint by taking average with surrounding 8 neighbors and hence the time in 
#hours that a new datapoint would be absent is the average of these neighbor values
#With this model we achieved a low accuracy of ~40% , which could be further improved by using a better classification technique
#Accuracy in this case could also be further improved by study correlation between attributes and dropping columns to prevent multi-collinearity

#The chosen value of K in this case is 8 because it had the lowest RMSE for all possible neighbor values (ranging from 1 to 30)
#The model was fitted for each such value of K and the resulting model RMSE was plotted versus the K value.
#We get a 'Elbow Curve' which has a local minima at 8, hence we chose the value as 8 for K in the KNN model, as it has the least RMSE.