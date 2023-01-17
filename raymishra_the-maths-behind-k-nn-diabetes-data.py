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
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# importing data

data=pd.read_csv('../input/diabetes.csv')

data.head()

y= data["Outcome"]

feature_data= data.drop("Outcome", axis= 1)

#split the dataset into X & y to perform standardization
scaled_features = StandardScaler().fit_transform(feature_data.values)

#scale the feature_data

scaled_data = pd.DataFrame(scaled_features, index=feature_data.index, columns=feature_data.columns)

#upon using standardization, we lose the columns and indexes, so we use pandas attributes to 

scaled_data.head()
train_scaled= pd.concat([scaled_data, y], axis=1)

train_scaled.head()
correlation = data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, annot=True)



plt.title('Correlation between different fearures')
# Defining a function to calculate euclidean distance between two data points

def EuclideanD(d1, d2, length):

    distance= 0

    for l in range(length):

        distance+=np.square(d1[l]-d2[l])

    return np.sqrt(distance)

        

    

    
def KNN(train, test, k):

    result= []

    length= test.shape[1]

    for y in range(len(test)):

        distances= {}

        sort= {}

        for x in range(len(train)):

            dist=EuclideanD(test.iloc[y],train.iloc[x],length) 

            '''

            basically we are iterating over the no. of features by calculating test.shape[1], and in EucliedeanD, we calculate distance 

            for the data point by using Euclidean formula for the respective features of a data point and then return sigma(distance)

            '''

            distances[x]=dist

            

            #sorting them on the basis of distance , which is the value in key-value pair of a dictionary

            sorted_d=sorted(distances.items(),key=lambda item: item[1])

            

        neighbours=[]

        #extract the top k neighbours, neighbours are the points which are at the smallest distances from the new test point

        for x in range(k):

            neighbours.append(sorted_d[x][0])

            

        majorityclassvotes={}

            

        #calculate the most frequent class among the k nearest neighbours

            

        for i in range(len(neighbours)):

            response= train.iloc[neighbours[i]][-1]

            if response in majorityclassvotes:

                majorityclassvotes[response]+= 1

            else:

                majorityclassvotes[response]= 1

        

        

        majorityvotesorted= sorted(majorityclassvotes.items(), key= lambda item: item[1], reverse= True)

        result.append(majorityvotesorted[0][0])

    return result

                    

    
import operator

dic= {}

dic= { 1: 1.2, 2: 1.56, 3: 5.2, 6: 7.1, 4: 2.7}

sorted(dic.items(), key=lambda item: item[1], reverse= True)
#Let's create a dummy set to test the model and we'll then compare it with sk-learn. 

test_data= [[0.42, 0.80, 0.25, 0.70, -0.12, 0.65, 0.36, 1.9], [-0.3, 0.50, 0.70, 0.20, -0.34, 0.86, 0.56, 2.8]]

test=pd.DataFrame(test_data)

# setting no. of neighbors

k=3

#Running our model

result=KNN(train_scaled,test,k)

print(result)
from sklearn.neighbors import KNeighborsClassifier

neigh=KNeighborsClassifier(n_neighbors=3)

neigh.fit(train_scaled.iloc[:,0:8],train_scaled['Outcome'])



print(neigh.predict(test))