!pip install category_encoders
# Importing Libraries

## For Data Operations and Visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

## For Classifiers
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
# Getting cwd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing Dataset
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
df.columns
# Dropping off redundant columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], inplace = True, axis = 1)  
df.info()
# Check for Imbalance
df.groupby('Exited')['Geography'].count()
l = LabelEncoder()
df['Gender'] = l.fit_transform(df['Gender'])
encoder = TargetEncoder()
df['country'] = encoder.fit_transform(df['Geography'], df['Exited'])
df.drop(['Geography'], inplace = True, axis = 1)
df
# Spliting into dependent and independent vectors
x = df.drop(['Exited'], axis = 1)
y = df.Exited
# Standard Scaling
S = StandardScaler()
x = S.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size = 0.25, 
                                                    random_state = 0)
# fitting my model
classifier = rfc(n_estimators = 100, random_state = 0, criterion = 'entropy')
classifier.fit(x_train, y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))
# fitting my model
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))
# fitting my model
classifier = GaussianNB()
classifier.fit(x_train, y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))
# fitting my model
classifier = MLPClassifier(activation = "relu", alpha = 0.05, random_state = 0)
classifier.fit(x_train, y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))
# Importing Necessary Libraries
from sklearn.ensemble import StackingClassifier
# Initialising the Stacking Algorithms
estimators = [
        ('naive-bayes', GaussianNB()),
        ('random-forest', rfc(n_estimators = 100, random_state = 0)),
        ('mlp', MLPClassifier(activation = "relu", alpha = 0.05, random_state = 0))
        ]
# Setting up the Meta-Classifier
clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(random_state = 0)
        )
# fitting my model
clf.fit(x_train, y_train)
# getting info about the hyperparameters 
clf.get_params()
# predicting the test set results
y_pred = clf.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))
# Defining Parameter Grid
params = {'final_estimator__C': [1.0,1.1,1.5],
          'final_estimator__max_iter': [50,100,150,200],
          'final_estimator__n_jobs': [1,-1,5],
          'final_estimator__penalty': ['l1','l2'],
          'final_estimator__random_state': [0],
          }
# Initialize GridSearchCV
grid = GridSearchCV(estimator = clf, 
                    param_grid = params, 
                    cv = 5,
                    scoring = "roc_auc",
                    verbose = 10,
                    n_jobs = -1)
# Fit GridSearchCV
grid.fit(x_train, y_train)
# predicting the test set results
y_pred = grid.predict(x_test)
# Checking Accuracy
print(classification_report(y_test, y_pred))