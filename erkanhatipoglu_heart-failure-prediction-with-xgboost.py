# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report, recall_score

import pandas_profiling
def calculate_cf (y_valid, preds):

    ''' A function that calculates the confusion matrix, accuracy, precision, recall, and f1_score.

        Accuracy, precision, recall, and f1_score can be easily obtained by using sklearn features. 

        this function is only for demontration purposes.'''

    

    # Calculating confusion_matrix

    CM = confusion_matrix(y_valid, preds)



    # Calculate True Positives(TP), False Positives(FP)

    # False Negative(FN) and True Negatives(TN) from confusion_matrix

    true_negatives = CM[0][0]

    false_negatives = CM[1][0]

    true_positives = CM[1][1]

    false_positives = CM[0][1]



    # You can easily get these values in sklearn using 

    # accuracy_score, precision_score, classification_report, etc.

    # I calculate these values for demonstration purposes.

    accuracy = (true_positives + true_negatives)/(true_positives + false_positives + false_negatives + true_negatives)

    precision = (true_positives) / (true_positives + false_positives)

    recall = (true_positives) / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)

    

    return true_negatives, false_negatives, true_positives, false_positives,accuracy, precision, recall, f1_score



def display_results (y_valid, preds):

    ''' A function that displays the results'''

    # get the results

    true_negatives, false_negatives, true_positives, false_positives,accuracy, precision, recall, f1_score = calculate_cf (y_valid, preds)

    

    blank= " "

    star = "*"

    print(blank*50 + "Death Event") 

    print(blank*30 + star*55)

    print(blank*35 + "Positive" + blank * 30 + "Negative")

    print(blank * 30 + star*18 + blank * 19 + star*18)

    print("Predicted" + blank*7 + "Positive" + blank  * 11 + str(true_positives)+ " (TP)" + blank * 32 + str(false_positives) + " (FP)")

    print("Class" + blank*11 + "Negative" + blank  * 11 + str(false_negatives)+ " (FN)" + blank * 31 + str(true_negatives) + " (TN)")

    print()

    print("Accuracy = (TP+TN)/(TP+FP+FN+TN) = {:.2f}".format(accuracy))

    print("Precision = (TP)/(TP+FP) = {:.2f}".format(precision))

    print("Recall = (TP)/(TP+FN) = {:.2f}".format(recall))

    print("F1 score = 2 * (Precision*Recall)/(Precision+Recall) = {:.2f}".format(f1_score))
# Read the data

train_data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')



# Make a copy to avoid changing original data

X = train_data.copy()



print('Data is OK')
X.head()
X.info()
X.describe()
# get the target, separate target and time from predictors

y = X.DEATH_EVENT              

X.drop(['DEATH_EVENT', 'time'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

# Display results

print ("Shapes:")

print ("X: {}".format(X.shape))

print ("y: {}".format(y.shape))

print()

print ("X_train: {}".format(X_train.shape))

print ("X_valid: {}".format(X_valid.shape))

print ("y_train: {}".format(y_train.shape))

print ("y_valid: {}\n".format(y_valid.shape))
# Select numeric columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

numerical_cols
# Define transformers

# Preprocessing for numerical data



numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[('numerical', numerical_transformer, numerical_cols)])
# Define Model

model = XGBClassifier(learning_rate = 0.1,

                            n_estimators=500,

                            max_depth=5,

                            min_child_weight=1,

                            gamma=0,

                            subsample=0.8,

                            colsample_bytree=0.8,

                            reg_alpha = 0,

                            reg_lambda = 1,

                            random_state=42)
# Preprocessing of validation data

X_valid_eval = preprocessor.fit(X_train, y_train).transform (X_valid)
# Display the number of remaining columns after transformation 

print("We have", X_valid_eval.shape[1], "features after transformation")
# Define XGBClassifier fitting parameters for the pipeline

fit_params = {"model__early_stopping_rounds": 50,

              "model__eval_set": [(X_valid_eval, y_valid)],

              "model__verbose": False,

              "model__eval_metric" : "error"}
# Create and Evaluate the Pipeline

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train, **fit_params)



# Get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = accuracy_score(y_valid,preds)



# Display the result

print("Score: {}".format(score))
# Display results

display_results (y_valid, preds)
# Define model parameters for grid search

param_grid = {'model__learning_rate': [0.1],

              'model__n_estimators': [13],

              'model__max_depth': [3, 4, 5, 6],

              'model__min_child_weight': [1, 2, 3, 4],

              'model__gamma': [0],

              'model__subsample': [0.60, 0.70, 0.80, 0.90],

              'model__colsample_bytree': [0.60, 0.70, 0.80, 0.90],

              'model__random_state' : [42]}
# Perform grid search

# Use model parameters defined above. We use scoring as recall to minimize 

# False Negatives

search = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1,scoring='recall')

search.fit(X, y)
# Get predictions

preds = search.predict(X_valid)



# Evaluate the model

score = accuracy_score(y_valid,preds)



print("Score: {}".format(score))
display_results (y_valid, preds)