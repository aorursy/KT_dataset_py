#Import all the necessary modules

#Import all the necessary modules

import pandas as pd

import numpy as np

import os

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
cardata = pd.read_csv("../input/carmpg/car-mpg (1).csv")
cardata.head()
cardata.dtypes
cardata.shape
cardata.info()
cardata.describe()
# Check for missing value

cardata.isna().sum()
# Na shows no missing value, but on careful data observation we could see "?" for hp values

cardata[cardata['hp']=="?"]
cardata['hp'].replace("?",np.nan, inplace=True)
# Now try to impute with mean of respective cylinders, but before this we must see the distribution for the variable

# We would drop na values and check distribution before taking call on whether imputation would be through mean, median 

import seaborn as sns

hp = cardata['hp'].dropna()

hp.count()
sns.distplot(pd.to_numeric(hp))
# Since this does not look to be normally distributed, let us impute by using median

cardata['hp'].fillna((cardata['hp'].median()), inplace=True)

cardata['hp'] = cardata['hp'].astype('float')
cardata.dtypes
cardata.corr(method='kendall')
sns.pairplot(cardata,diag_kind='kde')
# Observations

# From diagonal plots we can see origin has 3 points (evident from data)

# yr - shows two peaks majorily

# acc,mpg are nearly normal

# cyl and disp shows 3 clusters while wt shows 2



# from bivariate plots we can see that mps shows negative liner relationship with wt,hp and disp 

# (correlation too gives high negative correlation)

# Cyl too shows negative correlation with levels
cardata.groupby(cardata['cyl']).mean()
# Further dig into data shows max mpg is for 4 cylinders vehicles



# Origin as pointed earlier indicates production point so should be broken into dummy variables



# Year would be more effective if we can transorm this to calculate age of vehicle. This dataset was used in 1983 so we would 

# subtract year from 83 to get the age



# Other continuous variables should be checked for outliers and should be normlized using z-score
# Calculate age of vehicle

cardata['age'] = 83-cardata['yr']

cardata.head()
#Convert origing into dummy variables (This again is subjected to business knowledge. We might drop this variable as well

# Inclusion is more to demonstrate on how to use categorical data)



one_hot = pd.get_dummies(cardata['origin'])

one_hot = one_hot.add_prefix('origin_')



# merge in main data frame

cardata = cardata.join(one_hot)

cardata.head()
# Let us now remove duplicate/irrelevant columns



cars_new = cardata.drop(['yr','origin','car_name'], axis =1)



cars_new.head()
# Missing value check was done above and hp column was treated with median values

# Let us check for outliers now
sns.boxplot(data=cars_new)
# We could see some outliers for mpg,hp and acc

sns.boxplot(y=cars_new['mpg'])
sns.boxplot(y=cars_new['hp'])
sns.boxplot(y=cars_new['acc'])
# Let us take logaritmic transform for hp,mpg and acc 

# To remove outliers

cars_new['hp'] = np.log(cars_new['hp'])

cars_new['acc'] = np.log(cars_new['acc'])

cars_new['mpg'] = np.log(cars_new['mpg'])



cars_new.head()
sns.boxplot(data=cars_new)
# This looks better.

# Now let us try to scale the variables



#Apply z score for only the numeric variables



from scipy.stats import zscore



cars_new.dtypes

numeric_cols = cars_new.select_dtypes(include=[np.int64, np.float64]).columns

numeric_cols

cars_new[numeric_cols] =cars_new[numeric_cols].apply(zscore)

cars_new.head()
# Variables are mow scaled. Let us now try to create clusters



cluster_range = range(1,15)

cluster_errors = []

for num_clusters in cluster_range:

    clusters = KMeans(num_clusters, n_init = 5)

    clusters.fit(cars_new)

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append(clusters.inertia_)



clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})

clusters_df
from matplotlib import cm



plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# We could see the bend at 4, so let us create 4 custers



kmeans = KMeans(n_clusters=4, n_init = 5, random_state=12345)

kmeans.fit(cars_new)
# Check the number of data in each cluster



labels = kmeans.labels_

counts = np.bincount(labels[labels>=0])

print(counts)
# Distribution looks fine.



# let us check the centers in each group

centroids = kmeans.cluster_centers_

centroid_df = pd.DataFrame(centroids, columns = list(cars_new) )

centroid_df.transpose()
# Group 1 has highest values for mpg while 3rd has lowest

# Group 0 has max no of cylinders and 2 forms of lower cylinder values

# As seen in correlation and pairplot, Group 0 has highest values for hp,wt and displ

# Group 1 seems to be comprising of newest cars

# Group 3 and 0 seems to be originated at point 3, while 2 in 2nd point and 1 again at point 3
# Add cluster number to original cars data



predictions = kmeans.predict(cars_new)

predictions

cardata["group"] = predictions

cardata['group'] = cardata['group'].astype('category')

cardata.head()
# Visualize the centers



cars_new["group"] = predictions

cars_new.boxplot(by = 'group',  layout=(3,4), figsize=(15, 10))
# Group 0 is characterised by lower acc, comparitely old models, higher wt, hp but lowest mpg with origin at 1

# Group 1 -Highest mpg, lower wt and hp. Lower age limits suggest comparitevly newer cars. Origin looks more 2

# Group 2 - Origin mostly in location 2, lower deviation in wts, and hp so medain mpg and acceleration

# Group 3 - Again slighlty higher in wt origin code as 1. Better performance in terms of mpg
# Export the data into csv for any further analysis



from pandas import ExcelWriter

writer = ExcelWriter('groups.xls')

cardata.to_excel(writer,'Sheet1')

writer.save()
# We can try similar analysis for 3 grps as well to check if we get more clear distinction among groups