# Install XGBoost if not already installed

# !pip install xgboost
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import all libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Import the data into pandas dataframe, the data in the csv is seperated with ;, hence we need to mention it in deliminator 

# dataset = pd.read_csv('/kaggle/input/subprime-credit-score/MulticlassTrainData.csv', delimiter=';')

dataset = pd.read_csv('/kaggle/input/subprime-credit-score/MulticlassTrainData.csv', delimiter=';')
# Visualize the first 5 records 

dataset.head(5)
# Information related to the dataset does not contains which is the target variable. After analyzing the columns looks like the last column 'default_class' is the dependent (target) variable

dataset['default_class'].value_counts()
# There are 6 classes related to the credit score. We want to first analyze the data before we go ahead with prediction part.

# Describe function will not show the columns that has a non-numerical value, i.e. Strings

dataset.describe()
# Counting the frequency of the values in W column

dataset['W'].value_counts()
# Get the column names with their data types and store them in new variable

# The goal here is to get the column names along with their data types so that we can get a list of columns which needs to be dropped from dataset

coln_names = dataset.dtypes

col_name = dataset.columns

col_name = list(col_name)



number_of_columns = []

for i in range(len(coln_names)):

    if coln_names[i] == 'object':

        number_of_columns.append(col_name[i])

        

number_of_columns[:10]

dataset = dataset.drop(number_of_columns, axis=1)
dataset.head(5)

# For this experiment, I have dropped all the columns that has object type i.e a column which has non-numeric value

# Since it produces 40k total columns as features, kaggle notebook will run out of memory. 
# You can use this line to onehot encode all the columns that has non-numeric value

# Since I do not have any it will keep the number of features same as before except neglecting the last column which is target variable

ab = pd.get_dummies(dataset.iloc[:, :-1], drop_first=True)
ab.head(5)
# Visualize the number of null values for Default_flag column 

ab['Default_flag'].isnull().value_counts()
# Name of columns that has missing values

new_abc_cols = ['C','D','E','TRB','NOB','gc_final_score','VAL_App1_IdentityCheck_numprimar', 'VAL_App1_IdentityCheck_numactive',

                             'VAL_App1_IdentityCheck_levelofco','VAL_App1_IdentityCheck_levelofc0','CifasDetected', 'Segment', 'Number_of_All_Settled_Accounts',

                              'VAL_App1_IdentityCheck_numsharer', 'Age_of_Oldest_Account_All_Accoun', 'GC_C_S1_CL_027', 'GC_P_C_S1_CL_106','gc_age_of_applicant', 'GC_P_C_S1_CL_154', 'GC_P_C_S1_CL_112','External_Debt']
# get the names of columns that has missing values (i.e. Null values)

abc = ab.isnull().sum() # Check each columns has how many total null values   
# From above analysis we can see that column FWB and Default_flag has high number of missing values (more than 50% of the data)

# The best solution for these two columns will be to remove them from the dataset inorder to avoid their impact on the prediction

ab = ab.drop(['FWB', 'Default_flag'], axis=1)
# In order to treat missing values we need to check how manu unique values are present in that column along with their frequency to decide what approach to use like mean, median and mode

# Mean is average, Median is the middle value when all values are sorted, and mode gives us the most often appearing value

# For this problem I am using mode for all columns, since there are very few rows which has null values

for i in range(len(new_abc_cols)):

    if not new_abc_cols[i] in ['Default_flag', 'FWB']:

        print(ab[new_abc_cols[i]].value_counts())
# Handle all missing values using mode 

for i in range(len(new_abc_cols)):

    # print(new_abc_cols[i][0])

    if new_abc_cols[i] in ['C','D','E','TRB','NOB','gc_final_score','VAL_App1_IdentityCheck_numprimar', 'VAL_App1_IdentityCheck_numactive',

                             'VAL_App1_IdentityCheck_levelofco','VAL_App1_IdentityCheck_levelofc0','CifasDetected', 'Segment', 'Number_of_All_Settled_Accounts',

                              'VAL_App1_IdentityCheck_numsharer', 'Age_of_Oldest_Account_All_Accoun', 'GC_C_S1_CL_027', 'GC_P_C_S1_CL_106','gc_age_of_applicant', 'GC_P_C_S1_CL_154', 'GC_P_C_S1_CL_112','External_Debt']:

        ab[new_abc_cols[i]].fillna(ab[new_abc_cols[i]].mode()[0], inplace=True)

# ab is our final dataset



# Split the dependent and independent variables

y = dataset['default_class'].copy()

y = y.values
# Creating numpy array with independent variables

# X = ab.copy()

X = ab.iloc[:, 4:].values
# Use this code if you encounter memory error to free up some space



# import gc

# del dataset

# gc.collect()
# Create various classifiers to check which performs better comparatively



# Model-1 KNN

from sklearn.neighbors import KNeighborsClassifier

classifier_1 = KNeighborsClassifier(n_neighbors=5)



# Model-2 RandomForest

from sklearn.ensemble import RandomForestClassifier

classifier_2 = RandomForestClassifier(n_estimators=100, random_state=0)



# Model-3 XgBoost

from xgboost import XGBClassifier

classifier_3 = XGBClassifier()

# Split data into training and testing dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=0)
# Fitting all 3 models to our training data

classifier_1.fit(X_train, y_train)
# Fitting RandomForest Classifier

classifier_2.fit(X_train, y_train)
# Fitting XGBoost Classifier

classifier_3.fit(X_train, y_train)
# Predict variables for test dataset

y_pred_1 = classifier_1.predict(X_test)

y_pred_2 = classifier_2.predict(X_test)

y_pred_3 = classifier_3.predict(X_test)
# Create confusion matrix to evaluate model's performance

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred_1)

cm2 = confusion_matrix(y_test, y_pred_2)

cm3 = confusion_matrix(y_test, y_pred_3)
# Print all cm to analyze each model's accuracy

print("KNN Confusion Matrix")

print(cm1)

print("\n RandomForest Classifier Matrix")

print(cm2)

print('\n XGBoost Classifier Matrix')

print(cm3)
from sklearn.metrics import accuracy_score

print("Accuracy of Knn algorithm", accuracy_score(y_test, y_pred_1))

print("Accuracy of RandomForest algorithm", accuracy_score(y_test, y_pred_2))

print("Accuracy of XGBoost algorithm", accuracy_score(y_test, y_pred_3))