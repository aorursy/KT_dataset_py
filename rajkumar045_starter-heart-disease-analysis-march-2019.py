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
#1 Import Libraries

%reset -f

# 1.1 Call Numerical libraries

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

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

# http://scikit-learn.org/stable/modules/tree.html#tree

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

heart = pd.read_csv("../input/heart.csv")

heart.head(2)

heart.dtypes

heart.shape

heart.dtypes.value_counts()

heart.target.value_counts()
heart.isnull().sum().sum()
### Split into Test and Training Data

X_train, X_test, y_train, y_test = train_test_split(

        heart.drop('target', 1), 

        heart['target'], 

        test_size = 0.3, 

        random_state=10

        )
X_train.head(2)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
X_train.isnull().sum().sum()

X_test.isnull().sum().sum()
X_train['sum'] = X_train.sum(numeric_only = True, axis=1)
X_train.head()
X_test['sum'] = X_test.sum(numeric_only = True,axis=1)
X_test.head()
# 4.1 Assume that value of '0' in a cell implies missing feature

#     Transform train and test dataframes

#     replacing '0' with NaN

#     Use pd.replace()

tmp_train = X_train.replace(0, np.nan)

tmp_test = X_test.replace(0,np.nan)
# 4.2 Check if tmp_train is same as train or is a view

#     of train? That is check if tmp_train is a deep-copy



tmp_train is X_train                # False

tmp_train._is_view            # False
# 4.3 Check if 0 has been replaced by NaN

tmp_train.head(2)

tmp_test.head(2)
# 5. Feature 2 : For every row, how many features exist

#                that is are non-zero/not NaN.

#                Use pd.notna()

tmp_train.notna().head(1)

X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)
X_train.head(2)

X_test.head()

y_train.head(2)

y_test.head(2)
# 6. Similary create other statistical features

#    Feature 3

#    Pandas has a number of statistical functions

#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats



feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    X_train[i] = tmp_train.aggregate(i,  axis =1)

    X_test[i]  = tmp_test.aggregate(i,axis = 1)
feat

X_train.shape

X_test.shape
X_train.shape                # 212 X (21 13_2_6)

X_train.head(1)

X_test.shape                 # 91 X 21 (13 + 2+6)

X_test.head(2)
# Generate features using random projections

#     First stack train and test data, one upon another

tmp = pd.concat([X_train,X_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )





# 12.1

tmp.shape     # 303x21
#Transform tmp to numpy array

tmp = tmp.values

tmp.shape    #(303, 21)
# Create 4 random projections/columns

NUM_OF_COM = 5
# Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)
# 10.2 fit and transform the (original) dataset

#      Random Projections with desired number

#      of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])
rp[: 5, :  3]
# Create some column names for these columns

#      We will use them at the end of this code

rp_col_names = ["r" + str(i) for i in range(5)]

rp_col_names
if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp])
tmp.shape
X_train.shape

X = tmp[: X_train.shape[0], : ]

X.shape 
test = tmp[X_train.shape[0] :, : ]
test.shape
X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y_train,

                                                    test_size = 0.3)
X_train.shape
X_test.shape
y_test.shape
y_train.shape
# 24 Decision tree classification

# 24.1 Create an instance of class

clf = dt(min_samples_split = 5,

         min_samples_leaf= 5

        )

start = time.time()

# 24.2 Fit/train the object on training data

#      Build model

clf = clf.fit(X_train, y_train)

end = time.time()

(end-start)/60     
#  Use model to make predictions

classes = clf.predict(X_test)

#  Check accuracy

(classes == y_test).sum()/y_test.size   
clf.feature_importances_        
clf.feature_importances_.size
# 25. Instantiate RandomForest classifier

clf1_rf = rf(n_estimators=15)
# 25.1 Fit/train the object on training data

#      Build model



start = time.time()

clf1_rf = clf1_rf.fit(X_train, y_train)

end = time.time()

(end-start)/60    
# 25.2 Use model to make predictions

classes1_rf = clf1_rf.predict(X_test)
# 25.3 Check accuracy

(classes1_rf == y_test).sum()/y_test.size  