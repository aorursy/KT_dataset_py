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

# Load dataset

cc_apps=pd.read_csv('../input/uci-credit-approval-data-set/UCI_crx.csv', header=None)

# Inspect data

cc_apps.head()
# Print summary statistics

cc_apps_description = cc_apps.describe()

print(cc_apps_description)



print("\n")



# Print DataFrame information

cc_apps_info = cc_apps.info()

print(cc_apps_info)



print("\n")



# Inspect missing values in the dataset

cc_apps.tail(17)
# Import numpy

import numpy as np

# Inspect missing values in the dataset

print(cc_apps.tail(17))



# Replace the '?'s with NaN

cc_apps = cc_apps.replace('?', np.NaN)



# Inspect the missing values again

cc_apps.tail(17)
# Impute the missing values with mean imputation

cc_apps.fillna(cc_apps.mean(), inplace=True)



# Count the number of NaNs in the dataset to verify

cc_apps.isnull().sum()
# Iterate over each column of cc_apps

for col in cc_apps.columns:

    # Check if the column is of object type

    if cc_apps[col].dtype == 'object':

        # Impute with the most frequent value

        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])



# Count the number of NaNs in the dataset and print the counts to verify

cc_apps.isnull().sum()
# Import LabelEncoder

from sklearn.preprocessing import LabelEncoder



# Instantiate LabelEncoder

le=LabelEncoder()



# Iterate over all the values of each column and extract their dtypes

for col in cc_apps.columns.values:

    # Compare if the dtype is object

    if cc_apps[col].dtype=='object':

    # Use LabelEncoder to do the numeric transformation

        cc_apps[col]=le.fit_transform(cc_apps[col])
# Import train_test_split

from sklearn.model_selection import train_test_split



# Drop the features 11 and 13 and convert the DataFrame to a NumPy array

cc_apps = cc_apps.drop([11, 13], axis=1)

cc_apps = cc_apps.values



# Segregate features and labels into separate variables

X,y = cc_apps[:,0:12] , cc_apps[:,13]



# Split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,

                                y,

                                test_size=0.33,

                                random_state=42)
# Import MinMaxScaler

from sklearn.preprocessing import  MinMaxScaler



# Instantiate MinMaxScaler and use it to rescale X_train and X_test

scaler = MinMaxScaler(feature_range=(0,1))

rescaledX_train = scaler.fit_transform(X_train)

rescaledX_test = scaler.fit_transform(X_test)
# Import LogisticRegression

from sklearn.linear_model import LogisticRegression



# Instantiate a LogisticRegression classifier with default parameter values

logreg = LogisticRegression()



# Fit logreg to the train set

logreg.fit(rescaledX_train,y_train)
# Import confusion_matrix

from sklearn.metrics import confusion_matrix



# Use logreg to predict instances from the test set and store it

y_pred = logreg.predict(rescaledX_test)



# Get the accuracy score of logreg model and print it

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))



# Print the confusion matrix of the logreg model

print(confusion_matrix(y_test, y_pred))
# Import GridSearchCV

from sklearn.model_selection import GridSearchCV



# Define the grid of values for tol and max_iter

tol = [0.01, 0.001 and 0.0001]

max_iter = [100, 150, 200]



# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values

param_grid = dict(tol= tol, max_iter= max_iter)
# Instantiate GridSearchCV with the required parameters

grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)



# Use scaler to rescale X and assign it to rescaledX

rescaledX = scaler.fit_transform(X)



# Fit data to grid_model

grid_model_result = grid_model.fit(rescaledX, y)



# Summarize results

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))