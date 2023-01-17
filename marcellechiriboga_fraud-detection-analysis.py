# Import libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
# Load the data

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")



# Show the first 10 observations from the dataset to have a basic sense of the data

df.head(11)
# Print summary statistics of each feature from the dataframe

df.describe()
# Explore the features available in the dataframe

print(df.info())
# Count the occurrences of each category from the `Class` variable and print them

occ = df['Class'].value_counts()

print(occ)
# Print the ratio of fraud cases, being `0 = non-fraud` and `1 = fraud`

print(occ / len(df.index))
# Define a function to create a scatterplot of the data and labels

def plot_data(X, y):

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label = "Non-Fraud", alpha = 0.5, linewidth = 0.15)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label = "Fraud", alpha = 0.5, linewidth = 0.15, c = 'r')

    plt.legend()

    return plt.show()
# Create X and y from df

X = df.loc[:, df.columns != 'Class'].values

y = df.Class.values
# Plot the data by running the plot_data function on X and y

plot_data(X, y)
# Define the resampling method

method = SMOTE(kind = 'regular')



# Create the resampled feature set

X_resampled, y_resampled = method.fit_sample(X, y)



# Plot the resampled data

plot_data(X_resampled, y_resampled)
# Print the value_counts on the original labels y

print(pd.value_counts(pd.Series(y)))



# Print the resampled value_counts using SMOTE

print(pd.value_counts(pd.Series(y_resampled)))
# Define which resampling method and which ML model to use in the pipeline

resampling = SMOTE(kind = "borderline2")

model_lr = LogisticRegression()



# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model

pipeline_lr = Pipeline([('SMOTE', resampling), ('Logistic Regression', model_lr)])



# Using a pipeline

# Split your data X and y, into a training and a test set and fit the pipeline onto the training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data

pipeline_lr.fit(X_train, y_train)

predicted_lr = pipeline_lr.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_lr))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_lr))
# Apply Logistic Regression without SMOTE to compare the scores

model_lr.fit(X_train, y_train)

predicted_lr_no_smote = model_lr.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_lr_no_smote))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_lr_no_smote))
# Define the model as the random forest

model_rf = RandomForestClassifier(random_state = 5)



# Define the pipeline, tell it to combine SMOTE with the Random Forest model

pipeline_rf = Pipeline([('SMOTE', resampling), ('Random Forest', model_rf)])



# Fit the model to the training set

pipeline_rf.fit(X_train, y_train)



# Obtain predictions from the test data

predicted_rf = pipeline_rf.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_rf))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_rf))
# Apply Random Forest without SMOTE to compare the scores

model_rf.fit(X_train, y_train)

predicted_rf_no_smote = model_rf.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_rf_no_smote))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_rf_no_smote))
# GridSearchCV to find Random Forest optimal parameters



# Define the model to use

model_rf = RandomForestClassifier(random_state = 5)



# Define the pipeline, tell it to combine SMOTE with the Random Forest model

pipeline_rf = Pipeline([('SMOTE', resampling), ('RandomForest', model_rf)])



# Define the parameter sets to test

param_rf = {

    'RandomForest__n_estimators': [50, 100], 

    'RandomForest__max_features': ['auto', 'log2'], 

    'RandomForest__max_depth': [8, 12], 

    #'RandomForest__criterion': ['gini', 'entropy']

}



# Combine the parameter sets with the defined model

grid_search_rf = GridSearchCV(estimator = pipeline_rf, param_grid = param_rf, cv = 3, scoring = 'recall', n_jobs = -1)



# Fit the model to the training data and obtain best parameters

grid_search_rf.fit(X_train, y_train)

grid_search_rf.best_params_
# Define the model as the random forest with the optimal parameters found by GridSearchCV

best_model_rf = RandomForestClassifier(n_estimators = 50, 

                                       max_features = 'auto', 

                                       max_depth = 8, 

                                       #criterion = 'entropy',

                                       random_state = 5)



# Define the pipeline, tell it to combine SMOTE with the optimized Random Forest model

pipeline_rf = Pipeline([('SMOTE', resampling), ('RandomForest', best_model_rf)])



# Fit the model to the training set

pipeline_rf.fit(X_train, y_train)



# Obtain predictions from the test data

predicted_rf = pipeline_rf.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_rf))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_rf))
# Define the model as the XGBoost

model_xg_boost = XGBClassifier(random_state = 10)



# Define the pipeline, tell it to combine SMOTE with the XGBoost model

pipeline_xg_boost = Pipeline([('SMOTE', resampling), ('XGBoost', model_xg_boost)])



# Fit the model to the training set

pipeline_xg_boost.fit(X_train, y_train)



# Obtain predictions from the test data

predicted_xg_boost = pipeline_xg_boost.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_xg_boost))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_xg_boost))
# GridSearchCV to find XGBoost optimal parameters



# Define the model to use

model_xg_boost = XGBClassifier(random_state = 10)



# Define the pipeline, tell it to combine SMOTE with the XGBoost model

pipeline_xg_boost = Pipeline([('SMOTE', resampling), ('XGBoost', model_xg_boost)])



# Define the parameter sets to test

param_xg = { 

    'XGBoost__n_estimators': [50, 75, 100],

    'XGBoost__max_depth': [3, 6, 12]

}



# Find best parameters

grid_search_xg = GridSearchCV(estimator = pipeline_xg_boost, param_grid = param_xg, cv = 3, scoring = 'recall', n_jobs = -1)



# Fit the model to the training data and obtain best parameters

grid_search_xg.fit(X, y)

grid_search_xg.best_params_
# Initialize a XGBoost Classifier with the best parameters

best_model_xg_boost = XGBClassifier(n_estimators = 50, max_depth = 3, random_state = 10)



# Define the pipeline, tell it to combine SMOTE with the best XGBoost model

pipeline_best_xg_boost = Pipeline([('SMOTE', resampling), ('XGBoost', best_model_xg_boost)])



# Fit the model to the training set

pipeline_best_xg_boost.fit(X_train, y_train)



# Obtain predictions from the test data

predicted_best_xg_boost = pipeline_best_xg_boost.predict(X_test)



# Obtain the results from the classification report and confusion matrix

print('Classifcation report:\n', classification_report(y_test, predicted_best_xg_boost))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = predicted_best_xg_boost))