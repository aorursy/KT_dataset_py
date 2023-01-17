"""

Created on Sun Mar 17 20:53:24 2019



@author: vivek

"""

# Call data manipulation libraries

import pandas as pd

import numpy as np



#Feature creation libraries

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features



#For feature selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria



#Data processing

#caling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

#Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder



#Splitting data

from sklearn.model_selection import train_test_split



#Decision tree modeling

from sklearn.tree import  DecisionTreeClassifier as dt



# RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf



# Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns



# Misc

import os, time, gc

print(os.listdir("../input"))

# Set working directory to read file

os.chdir("../input")



# Read the file provided

heartdf=pd.read_csv("heart.csv")
# Have a look at the data

heartdf.shape

heartdf.dtypes # All afeatures are integers

heartdf.head(3)
#Seperate the predictors and target columns



x=heartdf.drop(columns = ['target'])#Predictors

x.shape  #(303, 13)

y=heartdf['target'] #target column

y.shape  #(303,)
x_train,x_test,y_train,y_test=train_test_split(     x,    # Predictors

                                                    y,                   # Target

                                                    test_size = 0.3      # split-ratio

                                                    )



x_train.shape #(212, 13)

x_test.shape #(91, 13)

y_train.shape #(212,)

y_test.shape  #(91,)
#check if all the columns are zeroes in any row

x = np.sum(x_train, axis = 1)

v = x.index[x == 0]             # Get index of the row which meets a condition

v



# Drop the rows from test data if v is not empty

x_test.drop(v, axis = 0, inplace = True)

x_test.shape  #no drop of rows           
# Check if there are Missing values? None

x_train.isnull().sum().sum()  # 0

x_test.isnull().sum().sum()   # 0
############################ BB. Feature Engineering #########################

############################ Using Statistical Numbers #####################
#  Assume that value of '0' in a cell implies missing feature

#     Transform train and test dataframes

#     replacing '0' with NaN

#     Use pd.replace()



#converting to nan for the quick calculations of mean,median,variance...

tmp_train = x_train.replace(0, np.nan)

tmp_test = x_test.replace(0,np.nan)





#  Check if 0 has been replaced by NaN

tmp_train.head(1)

tmp_test.head(1)



tmp_train.notna().head(1)

x_train["count_not0"] = tmp_train.notna().sum(axis = 1)

x_test['count_not0'] = tmp_test.notna().sum(axis = 1)

x_test['count_not0'].head()
# Similary create other statistical features

# Pandas has a number of statistical functions like  "var", "median", "mean", "std", "max", "min"

# Row sums of features 1:13. More successful when data is binary.

feat = [ "var", "median", "mean", "std", "max", "min","sum"]

for i in feat:

    x_train[i] = tmp_train.aggregate(i,  axis =1)

    x_test[i]  = tmp_test.aggregate(i,axis = 1)



x_train.shape # (212, 21)

x_train.head()
############################################################################

################ Feature creation Using Random Projections ##################

#  Random projection is a fast dimensionality reduction feature

#     Also used to look at the structure of data
#  Generate features using random projections

#     First stack train and test data, one upon another

tmp = pd.concat([x_train,x_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )

tmp.shape
#  Let us create 5 random projections/columns

#     This decision, at present, is arbitrary

NUM_OF_COM = 5



# Transform tmp t0 numpy array

#      Henceforth we will work with array only

tmp = tmp.values

tmp.shape    # (303, 21)
#  Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)
# fit and transform the (original) dataset

# Random Projections with desired number of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])

rp.shape
# Create some column names for these columns

#      We will use them at the end of this code

rp_col_names = ["r" + str(i) for i in range(5)]

rp_col_names
###############################################################################

############################ Feature creation using kmeans ####################

######################Can be skipped without loss of continuity################

# Before clustering, scale data

#  Create a StandardScaler instance

se = StandardScaler()

# fit() and transform() in one step

tmp = se.fit_transform(tmp)
#     No of centroids is no of classes in the 'target'

centers = heartdf['target'].nunique()    # 2 unique classes

centers               # 2
# Begin clustering

#  First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 2)         # Parallel jobs for n_init
# Next train the model on the original data only

kmeans.fit(tmp[:, : 13])



#  Get clusterlabel for each row (data-point)

kmeans.labels_

kmeans.labels_.size   # 303
# If Cluster labels are categorical convert them to dummy

# Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)
#  Use ohe to learn data

#      ohe.fit(kmeans.labels_)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()

                                          # '-1' is a placeholder for actual
# Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))#Reshape is used because onehot cannot recognise if it is in array form like (1,2,3,4,,,,)

dummy_clusterlabels

dummy_clusterlabels.shape    # 303 X 2 (as many as there are classes)
#  We will use the following as names of new nine columns

#      We need them at the end of this code



k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
############################ Interaction features #######################

# Will require lots of memory if we take large number of features

#     Best strategy is to consider only impt features



degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)#not includes f1square and f2square

                          include_bias = False    # No constant term

                          )
#  Consider only first 8 features to save memory

#      fit and transform

df =  poly.fit_transform(tmp[:, : 8])



df.shape     # 303 X 36
# Generate some names for these 36 columns

poly_names = [ "poly" + str(i)  for i in range(36)]

poly_names
################# concatenate all features now ##############################



# Append now all generated features together

# Append random projections, kmeans and polynomial features to tmp array



tmp.shape          # 303 X 21
# If variable, 'dummy_clusterlabels', exists, stack kmeans generated

# columns also else not. 'vars()'' is an inbuilt function in python.All python variables are contained in vars().



if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==





tmp.shape          # 303 X 64  
#  Separate train and test

X = tmp[: x_train.shape[0], : ]

X.shape                             # 212 X 64 if no kmeans: (61878, 126)



test = tmp[x_train.shape[0] :, : ]

test.shape                         # 91 X 64; if no kmeans: (144367, 126)
################## Model building #####################



y_train.shape

#  Split data into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y_train,

                                                    test_size = 0.3)

X_train.shape    # 148 X 64

X_test.shape     # 64 X 64;
# Decision tree classification

#  Create an instance of class

clf = dt(min_samples_split = 5,

         min_samples_leaf= 4

        )
#  Fit/train the object on training data

#      Build model

clf = clf.fit(X_train, y_train)

            

# Use model to make predictions

classes = clf.predict(X_test)



#  Check accuracy

(classes == y_test).sum()/y_test.size
#  Get feature importance

clf.feature_importances_        # Column-wise feature importance

clf.feature_importances_.size   # 64



# To our list of column names, append all other col names

#      generated by random projection, kmeans (onehotencoding)

#      and polynomial features

#      But first check if kmeans was used to generate features

colNames = x_train.columns.values

len(colNames)

if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined

    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

else:

    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==

# Create a dataframe of feature importance and corresponding

#      column names. Sort dataframe by importance of feature

feat_imp = pd.DataFrame({

                   "importance": clf.feature_importances_ ,

                   "featureNames" : colNames

                  }

                 ).sort_values(by = "importance", ascending=False)



feat_imp.shape                   # 135 X 2 ; without kmeans: (126,2)

feat_imp.head(30)
# Plot feature importance for first 20 features

g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)



#In below graph we can clearly observe that cluster K0 plays a mojor role in Decision tree classification
# Random Forest classification

#  Instantiate RandomForest classifier

clf = rf(n_estimators=50)



#  Fit/train the object on training data

#      Build model

clf = clf.fit(X_train, y_train)



#  Use model to make predictions

classes = clf.predict(X_test)

# Check accuracy

(classes == y_test).sum()/y_test.size  
#  Get feature importance

clf.feature_importances_        # Column-wise feature importance

clf.feature_importances_.size   # 64





# To our list of column names, append all other col names

#      generated by random projection, kmeans (onehotencoding)

#      and polynomial features

#      But first check if kmeans was used to generate features

colNames = x_train.columns.values

len(colNames)

if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined

    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

else:

    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==
feat_imp = pd.DataFrame({

                   "importance": clf.feature_importances_ ,

                   "featureNames" : colNames

                  }

                 ).sort_values(by = "importance", ascending=False)





feat_imp.shape                   # 135 X 2 ; without kmeans: (126,2)

feat_imp.head(30)
# Plot feature importance for first 20 features

g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)