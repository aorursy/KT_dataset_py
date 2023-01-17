# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.tree

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn import datasets

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics 

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import copy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/"+"train.csv")

print(train_data.columns.values,len(train_data))



test_data = pd.read_csv("../input/"+"test.csv")

#print(test_data.columns.values,len(test_data))

%matplotlib inline
# Let's train the data based on the decision tree:

row,column = train_data.shape

print(row,column)



#Train and test use sklearn:

# read header only:

feature_cols = list(train_data)[:-1]

print(feature_cols)



X_train = train_data[feature_cols]

#X_train = X_train.select_dtypes(exclude=['object'])



Y_train = train_data[list(train_data)[-1]]





X_test = test_data[feature_cols]







X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)



## X_train and X_test won't have the same numbers of columns since they may not have the same objects in each row. So let's make

# sure they have the same. numbers of columns(train) must be bigger than columns(test)

print(X_train.shape)

print(X_test.shape)



missing_cols = set( X_train.columns ) - set( X_test.columns )

print(missing_cols)

for c in missing_cols:

        X_test[c] = 0



# Then they have the same columns!

print(X_train.shape)

print(X_test.shape)

Id = list(X_test["Id"])

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)



X_test = my_imputer.fit_transform(X_test)

from sklearn.ensemble import ExtraTreesRegressor

forest = ExtraTreesRegressor(n_estimators=1000,n_jobs=-1)



forest.fit(X_train,Y_train)

importances = forest.feature_importances_



print(importances.shape)

mask_50 = importances>np.nanpercentile(importances,70)

plt.hist(importances[mask_50],bins=np.linspace(0,0.03,21))



## use mask 50 to select columns!!


# Create Decision Tree classifer object using random forest

clf = RandomForestRegressor(n_jobs=-1,n_estimators=100)



# Train Decision Tree Classifer

### Use one hot code since there are strings inside:

# deal with missing data



##optional Since we only have saleprice for the training set. Let's do a cross validation based on the training set: 7:3



clf = clf.fit(X_train[:,mask_50],Y_train)



#Predict the response for test dataset



plt.plot(Y_train,clf.predict(X_train[:,mask_50]),"ko")



Y_pred = clf.predict(X_test[:,mask_50])


df = pd.DataFrame({"Id":Id,list(train_data)[-1]:Y_pred})

# save it:

df.to_csv('csv_to_submit.csv', index = False)

print("Done")