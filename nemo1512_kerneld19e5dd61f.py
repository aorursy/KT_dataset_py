#Importing all the necessary packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import PowerTransformer
#First step in analysing any dataset is to check if it has any missing values. So let's do that first



data =pd.read_csv("../input/CC GENERAL.csv")

missing = data.isna().sum()

print(missing)
# Aha! Minimum payments and credit limit has missing values. We can fill in those missing values with either mean or median of its respective column.



data['MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median())

data['CREDIT_LIMIT'] = data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median())

data = data.drop(['CUST_ID'],axis=1)
# Let's take a look at how our data looks

data.head()
# It's always a good practice to remove unnecessary information from your dataset. This helps the algorithm converge better

# To find out which features are important, let's take a look their variance. Variance prvoides a quick overview of how spread the feature is. 

# Lower the varaince, less important the feature is.



for j in list(data.columns.values):

    print("Feature: {0}, Variance: {1}".format(j,data[j].var()))
# Let's get rid of the fatures with less variance



data = data.drop(["BALANCE_FREQUENCY","PURCHASES_FREQUENCY","ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_FREQUENCY","CASH_ADVANCE_TRX","PURCHASES_TRX","PRC_FULL_PAYMENT","TENURE"],axis=1)

for j in list(data.columns.values):

    print("Feature: {0}, Variance: {1}".format(j,data[j].var()))
# Next step is to chekc if we are dealing with a lot of outliers. Having a lot of them can cause K means to perform poorly.

plt.figure(figsize=(20,10))

for j in list(data.columns.values):

    plt.scatter(y=data[j],x=[i for i in range(len(data[j]))],s=[20])

plt.legend()
# Luckily, this dataset has very few outliers which we can ignore for now. However, one problem that is evident in the graph above is the scale of values

# Any machine learning model would perform better over a scaled dataset compared to a non -scaled one. So let's get that done.



X = PowerTransformer(method='yeo-johnson').fit_transform(data)
# Now that our data is ready, we can run our K-means algorithm over it. The trick here is to guess the number of clusters you want k-means to make.

# This is called the elbow technique



wcss = []

for ii in range( 1, 30 ):

    kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict(X)

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
# Somewhere near 7, the slope starts to flatten signifancty. So 7 is a good no of clusters to start with.





kmeans = KMeans(n_clusters=7, init="k-means++", n_init=10, max_iter=300) 

y_pred = kmeans.fit_predict(X)
# Now that we have our clusters, the only thing left to do is find out what are the common traits amongst the members in each cluster

# A pairplot is the simplest way to visualize this relationship.



data["cluster"] = y_pred

cols = list(data.columns)

ss = sns.pairplot( data[ cols ], hue="cluster")

plt.legend()