# Feature Enggineering and modelling on Heart Disease data base



import numpy as np 

import pandas as pd 





import os

print(os.listdir("../input"))

dataset=pd.read_csv("../input/heart.csv")



# Importing requiste libraries for modelling

from sklearn.random_projection import SparseRandomProjection as sr  

from sklearn.cluster import KMeans                    

from sklearn.preprocessing import PolynomialFeatures  

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import  DecisionTreeClassifier as dt

from sklearn.ensemble import RandomForestClassifier as rf

import matplotlib.pyplot as plt

import seaborn as sns

import os, time, gc
pd.options.display.max_columns = 300
dataset.head(2)
# Splitting and checking the split

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('target', 1), dataset['target'], test_size = 0.3, random_state=10)
X_train.shape
X_test.shape 
y_train.shape
X_train.isnull().sum().sum() # no missing values
y_test.isnull().sum().sum() # no missing values
#Feature Engineering
X_train['sum'] = X_train.sum(numeric_only = True, axis=1)

X_test['sum'] = X_test.sum(numeric_only = True,axis=1)
#Replace missing feature with NaN
tmp_train = X_train.replace(0, np.nan)

tmp_test = X_test.replace(0,np.nan)
# Estimating the no. of Features

X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)
#Creating other statistical Features

feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    X_train[i] = tmp_train.aggregate(i,  axis =1) #create columns of feat

    X_test[i]  = tmp_test.aggregate(i,axis = 1)
del(tmp_train)

del(tmp_test)

gc.collect()
# Seperating the target fetaure and storing column names

target = y_train

colNames = X_train.columns.values
#Feature creation Using Random Projections

tmp = pd.concat([X_train,X_test],

                axis = 0,            

                ignore_index = True

                )
# Creating numpy arrays for further analysis

tmp = tmp.values

tmp.shape # (303, 21)
# Working with 7 projection columns and creating instance of a class and fitting the data



NUM_OF_COM = 7

rp_instance = sr(n_components = NUM_OF_COM)

rp = rp_instance.fit_transform(tmp[:, :13])
rp_col_names = ["r" + str(i) for i in range(10)]

rp_col_names
# K Means : Feature Creation

se = StandardScaler()



tmp = se.fit_transform(tmp)



tmp.shape
centers = target.nunique()    # 2 unique classes

centers
kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 2)         # Parallel jobs for n_init







# Training the model on the original data 

kmeans.fit(tmp[:, : 13])
kmeans.labels_

kmeans.labels_.size 
#  Converting Cluster labels from categorical  to dummy.



# Creating an instance of OneHotEncoder class, learn and transform data

ohe = OneHotEncoder(sparse = False)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()

                                          # '-1' is a placeholder for actual

# 19.3 Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape 

# Giving Column names to the Features by K means 

k_means_names = ["k" + str(i) for i in range(9)]

k_means_names
#To select the pertinent Features for optimizing memory usage

degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )
#Using only the first 5 columns and generate names for these columns

df =  poly.fit_transform(tmp[:, : 5])

poly_names = [ "poly" + str(i)  for i in range(15)]

# Bringing in all the Features together

tmp.shape # (303, 21)

tmp = np.hstack([tmp,rp, df]) # without K Means Features
#  Separating train and test

X = tmp[: X_train.shape[0], : ]

X.shape # (212, 43)





test = tmp[X_train.shape[0] :, : ]

test.shape  #(91, 43)

 
y = pd.concat([y_train,y_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )

y.shape #(303,)
# to be completed.