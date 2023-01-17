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


# Load Data from teh CSV

data=pd.read_csv("../input/heart.csv")



# Importing libraries for modelling

from sklearn.random_projection import SparseRandomProjection as sr  

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures   # Interactions

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







data.shape
data.head(2)
data.dtypes.value_counts() 
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = 0.3, random_state=10)

X_train.shape
y_train.shape
X_test.shape 
y_test.shape
X_train.isnull().sum().sum()
X_test.isnull().sum().sum() 
X_train['sum'] = X_train.sum(numeric_only = True, axis=1) # training set

X_test['sum'] = X_test.sum(numeric_only = True,axis=1) # testing set
tmp_train = X_train.replace(0, np.nan) # Replace missing features with NaN

tmp_test = X_test.replace(0,np.nan)

tmp_train.head(2)
tmp_train._is_view   # To find out if its a view or copy
X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)
X_train.head(1)
colNames = X_train.columns.values
colNames
tmp = pd.concat([X_train,X_test],  axis = 0, ignore_index = True)

tmp.shape
target = y_train

colNames = X_train.columns.values
tmp = tmp.values

# Working with 5 projection columns and creating instance of a class and fitting the data



NUM_OF_COM = 5

rp_instance = sr(n_components = NUM_OF_COM)

rp = rp_instance.fit_transform(tmp[:, :13])

rp[: 2, :  3]

rp_col_names = ["r" + str(i) for i in range(5)]

rp_col_names
centers = y_train.nunique()    # number of unique elements in the group

centers
kmeans = KMeans(n_clusters=centers, n_jobs = 2)   

kmeans.fit(tmp[:, : 13])

kmeans.labels_
kmeans.labels_.size 
# select the pertinent Features for optimizing memory usage

degree = 3

poly = PolynomialFeatures(degree,                 # Degree 3

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )


df =  poly.fit_transform(tmp[:, : 3])

poly_names = [ "poly" + str(i)  for i in range(15)]
del tmp

gc.collect()