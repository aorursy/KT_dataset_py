# Import pandas

# ... YOUR CODE FOR TASK 1 ...

import pandas as pd



# Load dataset

cc_apps = pd.read_csv("../input/cc_approvals.csv", header=None)



# Inspect data

# ... YOUR CODE FOR TASK 1 ...

print(cc_apps.head())

cc_apps
# Print summary statistics

cc_apps_description = cc_apps.describe()

print(cc_apps_description)



print("\n")



# Print DataFrame information

cc_apps_info = cc_apps.info()

print(cc_apps_info)



print("\n")



# Inspect missing values in the dataset

# ... YOUR CODE FOR TASK 2 ...

cc_apps.tail(17)
# Import numpy

# ... YOUR CODE FOR TASK 3 ...

import numpy as np



# Inspect missing values in the dataset

print(cc_apps.isnull().values.sum())



# Replace the '?'s with NaN

cc_apps = cc_apps.replace("?",np.NaN)



# Inspect the missing values again

# ... YOUR CODE FOR TASK 3 ...

cc_apps.tail(17)
# Impute the missing values with mean imputation

cc_apps = cc_apps.fillna(cc_apps.mean())



# Count the number of NaNs in the dataset to verify

# ... YOUR CODE FOR TASK 4 ...

print(cc_apps.isnull().values.sum())
# Iterate over each column of cc_apps

print(cc_apps.info())



for col in cc_apps.columns:

    # Check if the column is of object type

    if cc_apps[col].dtypes == 'object':

        # Impute with the most frequent value

        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].value_counts().index[0])



# Count the number of NaNs in the dataset and print the counts to verify

# ... YOUR CODE FOR TASK 5 ...

print(cc_apps.isnull().values.sum())
# Import LabelEncoder

# ... YOUR CODE FOR TASK 6 ...

from sklearn.preprocessing import LabelEncoder



# Instantiate LabelEncoder

# ... YOUR CODE FOR TASK 6 ...

le = LabelEncoder()



# Iterate over all the values of each column and extract their dtypes

for col in cc_apps.columns:

    # Compare if the dtype is object

    if cc_apps[col].dtype=='object':

    # Use LabelEncoder to do the numeric transformation

        cc_apps[col]=le.fit_transform(cc_apps[col])
cc_apps
# Import MinMaxScaler

# ... YOUR CODE FOR TASK 7 ...

from sklearn.preprocessing import MinMaxScaler



# Drop features 10 and 13 and convert the DataFrame to a NumPy array

cc_apps = cc_apps.drop([cc_apps.columns[10],cc_apps.columns[13]], axis=1)

cc_apps
cc_apps = cc_apps.values



# Segregate features and labels into separate variables

X,y = cc_apps[:,0:13], cc_apps[:,13]





# Instantiate MinMaxScaler and use it to rescale

scaler = MinMaxScaler(feature_range=(0,1))

rescaledX = scaler.fit_transform(X)

# Import train_test_split

# ... YOUR CODE FOR TASK 8 ...

from sklearn.model_selection import train_test_split



# Split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(rescaledX,

                                                    y,

                                                    test_size=0.33,

                                                    random_state=42)
# Import LogisticRegression

# ... YOUR CODE FOR TASK 9 ...

from sklearn.linear_model import LogisticRegression



# Instantiate a LogisticRegression classifier with default parameter values

logreg = LogisticRegression()



# Fit logreg to the train set

# ... YOUR CODE FOR TASK 9 ...

logreg.fit(X_train,y_train)
# Import confusion_matrix

# ... YOUR CODE FOR TASK 10 ...

from sklearn.metrics import confusion_matrix



# Use logreg to predict instances from the test set and store it

y_pred = logreg.predict(X_test)



# Get the accuracy score of logreg model and print it

print("Accuracy of logistic regression classifier: ", logreg.score(X_test, y_test))



# Print the confusion matrix of the logreg model

# ... YOUR CODE FOR TASK 10 ...

confusion_matrix(y_test, y_pred)
# Import GridSearchCV

# ... YOUR CODE FOR TASK 11 ...

from sklearn.model_selection import GridSearchCV



# Define the grid of values for tol and max_iter

tol = [0.01, 0.001, 0.0001]

max_iter = [100, 150, 200]



# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values

param_grid = dict(tol=tol, max_iter=max_iter)

print(param_grid)
# Instantiate GridSearchCV with the required parameters

grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)



# Fit data to grid_model

grid_model_result = grid_model.fit(rescaledX, y)



# Summarize results

best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))