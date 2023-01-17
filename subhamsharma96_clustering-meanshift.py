import numpy as np 

import pandas as pd 
titanic_data = pd.read_csv('../input/train.csv')

titanic_data.head()
titanic_data.drop(['PassengerId','Name','Ticket','Cabin'],'columns',inplace=True)

titanic_data.head()
from sklearn import preprocessing



le=preprocessing.LabelEncoder()

titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))

titanic_data.head()
titanic_data = pd.get_dummies(titanic_data,columns=['Embarked'])

titanic_data.head()
titanic_data[titanic_data.isnull().any(axis=1)]
titanic_data = titanic_data.dropna()
from sklearn.cluster import MeanShift



analyzer = MeanShift(bandwidth=30) #We will provide only bandwith in hyperparameter . The smaller values of bandwith result in tall skinny kernels & larger values result in short fat kernels.

#We found the bandwith using the estimate_bandiwth function mentioned in below cell.

analyzer.fit(titanic_data)
#Below is a helper function to help estimate a good value for bandwith based on the data.

"""from sklearn.cluster import estimate_bandwith

estimate_bandwith(titanic_data)"""   #This runs in quadratic time hence take a long time

labels = analyzer.labels_
np.unique(labels)
#We will add a new column in dataset which shows the cluster the data of a particular row belongs to.

titanic_data['cluster_group'] = np.nan

data_length=len(titanic_data)

for i in range(data_length):

    titanic_data.iloc[i,titanic_data.columns.get_loc('cluster_group')] = labels[i]

titanic_data.head()

titanic_data.describe()
#Grouping passengers by Cluster

titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()

#Count of passengers in each cluster

titanic_cluster_data['Counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())

titanic_cluster_data