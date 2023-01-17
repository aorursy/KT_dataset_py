# Goal : Create Model for Prediction of heart disease using feature engineering



# Steps followed

# 1) Introduce the data

# 2) Exploration

# 3) Data cleaning

# 4) Feature engineering

# 5) Model training

# 

# How it is done

#         i)   using pandas and sklearn for modeling

#         ii)  Feature engineering

#                   a) Using statistical measures

#                   b) Using Random Projections

#                   c) Using clustering

#                   d) USing interaction variables

#        iii)  Feature selection

#                   a) Using derived feature importance from modeling

#                   b) Using sklearn FeatureSelection Classes

#         iv)  One hot encoding of categorical variables

#          v)  Classifciation using Decision Tree and RandomForest
# 1.0 Clear memory

%reset -f



# 1.1 Call data manipulation libraries

import pandas as pd

import numpy as np



# 1.2 Feature creation libraries

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features



# 1.3 For feature selection

# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria



# 1.4 Data processing

# 1.4.1 Scaling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

# 1.4.2 Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder



# 1.5 Splitting data

from sklearn.model_selection import train_test_split



# 1.6 Decision tree modeling

from sklearn.tree import  DecisionTreeClassifier as dt



# 1.7 RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf



# 1.8 Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns



# 1.9 Misc

import os, time, gc
# 2.0 Set working directory and read file

print(os.listdir("../input"))

# 2.1 Read heart.csv file

heart_data = pd.read_csv("../input/heart.csv")
#2.2 understand the heart data

heart_data.shape     

heart_data.head(2)

heart_data.dtypes.value_counts()
#  303 observation and 14 features

#  Only Continous numeric data and no categorical

#  13 heart related features and 1 target column

# Let us drop the target column from the dataframe before working on features



target = heart_data['target']

target.tail(2)



heart_data.drop(columns = ['target'], inplace = True)

heart_data.shape  



heart_data.describe()
# 3 Check if there are Missing values?

heart_data.isnull().sum().sum() 
# 4 Let us see graphs for age and Cholestrol before we proceed

age = heart_data.loc[ : , 'age']

chol = heart_data.loc[ :, 'chol']
plt.figure()

plt.title('Age vs cholestrol')

plt.xlabel ('Age')

plt.ylabel ('Cholesterol')

plt.plot(age,chol)

plt.show()
# 5 add more derived features using sum, count, var, median, mean, std, max, min

heart_data['sum'] = heart_data.sum(numeric_only = True, axis=1)





feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    heart_data[i] = heart_data.aggregate(i,  axis =1)
heart_data.shape
# store column names of the data

colNames = heart_data.columns.values

colNames
# 6 Transform to array before applying random projection

tmp = heart_data.values



# create 4 random projection column

NUM_OF_COM = 4



# Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)



#  fit and transform the (original) dataset

#  Random Projections with desired number of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])
# Look at some features

rp[: 5, :  3]  
# create some column names

rp_col_names = ["r" + str(i) for i in range(5)]

rp_col_names
# 7 Before clustering, scale data for unit variance

# Create a StandardScaler instance

se = StandardScaler()

# fit and transform() in one step

tmp = se.fit_transform(tmp)

tmp.shape
# No of centroids is no of classes in the 'target'

centers = target.nunique()    # 2 unique classes

centers             
# 8 Begin clustering



# First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 2)         # Parallel jobs for n_init





# Next train the model on the original data only

kmeans.fit(tmp[:, : 13])



kmeans.labels_

kmeans.labels_.size
# Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)



# Use ohe to learn data

#  ohe.fit(kmeans.labels_)

ohe.fit(kmeans.labels_.reshape(-1,1))     



# Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape 
# create some more column names

k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
# Generate polynomial and interaction features



degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )



df =  poly.fit_transform(tmp[:, : 5])



df.shape

# Create polynomial columns

poly_names = [ "poly" + str(i)  for i in range(15)]

poly_names
# check the shape of the data 

tmp.shape
if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df]) 
tmp.shape
X = tmp[: heart_data.shape[0], : ]

X.shape
# No longer need tmp as entering next stage to split the data for training and test

del tmp

gc.collect() 
# Split into training and test data



X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    target,

                                                    test_size = 0.3)



X_train.shape

X_test.shape 
# Decision tree classification

clf = dt(min_samples_split = 5,

         min_samples_leaf= 5

        )



clf = clf.fit(X_train, y_train)



classes = clf.predict(X_test)



(classes == y_test).sum()/y_test.size  # Check the accuracy
# Found 73.6% accuracy when the model applied on the test set
# Random forest classifier

clf = rf(n_estimators=50)



clf = clf.fit(X_train, y_train)



classes = clf.predict(X_test)



(classes == y_test).sum()/y_test.size  # Check the accuracy
# Random forest model achieved higher accuracy of 82.4% when applied on the test set
clf.feature_importances_        # Column-wise feature importance

clf.feature_importances_.size
# Compute confusion matrix to evaluate the metrics of the classification



from sklearn.metrics import confusion_matrix
f  = confusion_matrix(classes, y_test )

f
tn, fp, fn, tp = f.ravel()
precision = tp/(tp+fp)

precision             
recall = tp/(tp + fn)

recall       