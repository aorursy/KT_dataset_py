import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import  DecisionTreeClassifier as dt

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.ensemble import RandomForestClassifier as rf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import export_graphviz #plot tree

import os, time, gc

data = pd.read_csv("../input/HeartRate.csv") #Loading of Data

data.head(2)

data.shape
data.dtypes.value_counts() 

data.isnull().sum().sum()  # 0
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = 0.3, random_state=10)
X_train.shape  

X_test.shape 

y_train.shape 

y_test.shape
X_train.isnull().sum().sum()  
X_test.isnull().sum().sum() 
X_train['sum'] = X_train.sum(numeric_only = True, axis=1) 

X_test['sum'] = X_test.sum(numeric_only = True,axis=1)

tmp_train = X_train.replace(0, np.nan)

tmp_test = X_test.replace(0,np.nan)
tmp_train is X_train
tmp_train._is_view  

tmp_train.head(2)
tmp_train.notna().head(1)
X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)

X_train.shape
feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    X_train[i] = tmp_train.aggregate(i,  axis =1)

    X_test[i]  = tmp_test.aggregate(i,axis = 1)

del(tmp_train)

del(tmp_test)
gc.collect()
X_train.shape  
X_train.head(1)

colNames = X_train.columns.values
colNames
tmp = pd.concat([X_train,X_test],  axis = 0, ignore_index = True)
tmp.shape
tmp = tmp.values
tmp.shape 
NUM_OF_COM = 5

rp_instance = sr(n_components = NUM_OF_COM)

rp = rp_instance.fit_transform(tmp[:, :13])

rp[: 2, :  3]
rp_col_names = ["r" + str(i) for i in range(5)]
rp_col_names
centers = y_train.nunique()  
centers   

kmeans = KMeans(n_clusters=centers, n_jobs = 2)   

kmeans.fit(tmp[:, : 13])

kmeans.labels_
kmeans.labels_.size 
ohe = OneHotEncoder(sparse = False)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels
dummy_clusterlabels.shape    #(303,2)
k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
degree = 2

poly = PolynomialFeatures(degree, interaction_only=True, include_bias = False)

df =  poly.fit_transform(tmp[:, : 5])

df.shape     # 303 X 15

poly_names = [ "poly" + str(i)  for i in range(15)]

poly_names
if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df]) 

    
tmp.shape  
X = tmp[: X_train.shape[0], : ]

X.shape
test = tmp[X_train.shape[0] :, : ]

test.shape
del tmp

gc.collect()
X_train, X_test, y_train, y_test = train_test_split(X,y_train,test_size = 0.3)
X_train.shape

clf = rf(n_estimators=50)

clf = clf.fit(X_train, y_train)

classes = clf.predict(X_test)

(classes == y_test).sum()/y_test.size 
clf.feature_importances_ 
clf.feature_importances_.size
if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined

    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

else:

    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==

len(colNames)
feat_imp = pd.DataFrame({ "importance": clf.feature_importances_ , "featureNames" : colNames } ).sort_values(by = "importance", ascending=False)

colNames
g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])