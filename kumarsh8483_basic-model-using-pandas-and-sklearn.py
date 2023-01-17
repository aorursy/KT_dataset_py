# Import Libradries 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import MeanShift

from sklearn.datasets.samples_generator import make_blobs

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import style

style.use("ggplot")

from sklearn import preprocessing, cross_validation

# Ensures graphs to be displayed in ipynb

%matplotlib inline   
# read data into dataframe

titanic_df = pd.read_csv('../input/train.csv',header=0)  # Always use header=0 to read header of csv files



orig_df = pd.DataFrame.copy(titanic_df)

titanic_df.head()

orig_df.info()

#dropping irrelevant column for building model

titanic_df.drop(['Name','Sex','Ticket'],1, inplace= True)

titanic_df.convert_objects(convert_numeric = True)

titanic_df.fillna(0, inplace= True)



print (titanic_df.head())
#handle all non-numeric data

def handle_non_numerical_data(titanic_df):

	columns = titanic_df.columns.values

	

	for column in columns:

		text_digit_vals = {}

		def convert_to_int(val):

			return text_digit_vals[val]

			

		if titanic_df[column].dtype !=np.int64 and titanic_df[column].dtype !=np.float64:

			column_contents = titanic_df[column].values.tolist()

			unique_elements = set(column_contents)

			x = 0

			for unique in unique_elements:

				if unique not in text_digit_vals:

					text_digit_vals[unique] = x

					x+=1

                    

			#map the new "id" value to replace the string

			titanic_df[column] = list(map(convert_to_int, titanic_df[column]))

			

	return titanic_df

	

titanic_df = handle_non_numerical_data(titanic_df)

#print (titanic_df.head())





X = np.array(titanic_df.drop(['Survived'],1). astype(float))

X = preprocessing.scale(X)

y = np.array(titanic_df['Survived'])



#create the fit using Meanshift

clf = MeanShift()

clf.fit(X)



#get various attributes from dataframes

labels = clf.labels_

cluster_centers = clf.cluster_centers_



#create new column for identfying clustergroup based on meanshift model

orig_df['cluster_group'] = np.nan



#iterate through various labels and populate to empty columns

for i in range(len(X)):

    orig_df['cluster_group'].iloc[i] = labels[i]



#check clusters with survival rates    

n_clusters_ =len(np.unique(labels))

survival_rates = {}





for i in range(n_clusters_):

    temp_df = orig_df[(orig_df['cluster_group']==float(i))]

    survival_cluster = temp_df[(temp_df['Survived'] ==1)]

    survival_rate = len(survival_cluster)/len(temp_df)

    survival_rates[i] = survival_rate

    

print (survival_rates)





submission = pd.DataFrame({

        "PassengerId":orig_df['PassengerId'],

        "Survived":orig_df['Survived'],

        "clustergroup":orig_df['cluster_group']

    })

submission.to_csv('titanic.csv',index=False)
#I got 5 clusters, you can check the survival rates of each of the cluster 

#Check for survival rates under each cluster

print(orig_df[ (orig_df['cluster_group']==1) ])

print(orig_df[ (orig_df['cluster_group']==2) ])

print(orig_df[ (orig_df['cluster_group']==3) ])

#print(orig_df[ (orig_df['cluster_group']==4) ])

#print(orig_df[ (orig_df['cluster_group']==5) ])

## Now that we completed our Basic Model we need to plot the learning curves which will help in improving the model