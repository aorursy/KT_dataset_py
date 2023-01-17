# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # Linear algebra

import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pf # Generates profile reports from a pandas DataFrame

import seaborn as sns # data visualization library based on matplotlib

from sklearn import preprocessing # provides several common utility functions and transformer classes to change raw feature vectors into a representation more suitable for the downstream estimators

from sklearn.model_selection import train_test_split # split arrays or matrices into random train and test subsets

from sklearn.linear_model import LogisticRegression # Logistic Regression (aka logit, MaxEnt) classifier

from sklearn.feature_selection import RFE # Feature ranking with recursive feature elimination.

from sklearn import metrics #  includes score functions, performance metrics and pairwise metrics and distance computations

from sklearn.metrics import classification_report, confusion_matrix # Build a text report showing the main classification metrics,Compute confusion matrix to evaluate the accuracy of a classification. 

from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score,accuracy_score

import matplotlib.pyplot as plt # is a state-based interface to matplotlib. It provides a MATLAB-like way of plotting

import math

%matplotlib inline 

# a magic function which sets the backend of matplotlib to the 'inline' backend

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Adjust pandas display and formatting settings



# Remove scientific notations and display numbers with 2 decimal points instead

pd.options.display.float_format = '{:,.3f}'.format        



# Increase cell width

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:98% !important; }</style>"))



# Update default style and size of charts

plt.style.use('ggplot')



# Increase max number of rows and columns to display in pandas tables

pd.set_option('display.max_columns', 500)           

pd.set_option('display.max_rows', 500) 
# load the data 

dfPersonalLoanData = pd.read_csv("../input/bank-personal-loan-modelling/Bank_Personal_Loan_Modelling.csv")

dfPersonalLoanData.head(10)

# Checking number of raws and columns

dfPersonalLoanData.shape
# Check field types

dfPersonalLoanData.dtypes
dfPersonalLoanData.describe().T
# Check for null values

print(dfPersonalLoanData.isnull().sum())



# check for na

print("\n")

print("*****check for na***** \n \n",  dfPersonalLoanData.isna().sum())
# Number of unique in column(s)

dfPersonalLoanData.nunique()
# Number of people with zero mortgage

print ("Number of people with zero mortgage:", (dfPersonalLoanData['Mortgage']==0).sum())

print("Number of people with zero credit card spending per month:", (dfPersonalLoanData['CCAvg']==0).sum())

# value counts for all categorical fields

for col in ['Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']:

    print('\nColumn:', col)         # "\n" indicates new line

    print(dfPersonalLoanData[col].value_counts())
sns.pairplot(dfPersonalLoanData[['Mortgage', 'Income', 'CCAvg', 'Personal Loan']], hue = 'Personal Loan', diag_kind = 'kde');
corr = dfPersonalLoanData.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
dfPersonalLoanData.groupby(dfPersonalLoanData['Personal Loan']).mean()



dfPersonalLoanData.head()
# Let's drop Experience which have some negative values, ID and Zip Code

dfPersonalLoanData.drop(columns ='ID',inplace=True)

dfPersonalLoanData.drop(columns='Education', inplace=True)

dfPersonalLoanData.drop(columns='Family', inplace=True)

dfPersonalLoanData.drop(columns ='Experience',inplace= True)

dfPersonalLoanData.drop(columns ='ZIP Code',inplace= True)
dfPersonalLoanData['Personal Loan'].value_counts(normalize=True) 
# Storing predictors and target in X and y variables





X = dfPersonalLoanData.drop('Personal Loan', axis=1)

y= dfPersonalLoanData['Personal Loan'] # target variable



# Split the data into train and test





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Creating logistic regression model



model = LogisticRegression(solver = 'liblinear')

model.fit(X_train, y_train)



y_predicted = model.predict(X_test)



# Evaluate mode performance 

print(classification_report(y_test, y_predicted))
# draw confusion metrix

def draw_cm(actual,predicted):

    cm = confusion_matrix(actual,predicted)

    sns.heatmap(cm,annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])

    plt.ylabel('observed')

    plt.xlabel('Predicted')

    plt.show()



draw_cm(y_test, y_predicted)
print('Accuracy on train set: {:.2f}'.format(model.score(X_train, y_train)))

print('Accuracy on test set: {:.2f}'.format(model.score(X_test, y_test)))

print('Recall score: {:.2f}'.format(recall_score(y_test,y_predicted)))

print('ROC AUC score: {:.2f}'.format(roc_auc_score(y_test,y_predicted)))

print('Precision score: {:.2f}'.format(precision_score(y_test,y_predicted)))
 #!pip install yellowbrick
from yellowbrick.classifier import ClassificationReport, ROCAUC

viz = ClassificationReport(LogisticRegression(solver = 'liblinear',random_state=42))

viz.fit(X_train,y_train)

viz.score(X_test,y_test)

viz.show()
# Improve the model using GridSearchCV

# model paramters 

model.get_params()

from sklearn.model_selection import GridSearchCV

param_grid = [{'solver': ['newton-cg','lbfgs','liblinear','sag','saga'], 'C': [0.001,0.01,0.1,0.25,0.5,0.75,1],

              'class_weight':['balanced'], 'penalty':['l2']}]

grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5, verbose=0)

grid_search.fit(X_train,y_train)

grid_search.best_estimator_
#new model

model_new = LogisticRegression(C=0.25, class_weight='balanced', dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=42, solver='newton-cg', tol=0.0001, verbose=0,

                   warm_start=False)



model_new.fit(X_train, y_train)



predicted = model_new.predict(X_test)



# Evaluate mode performance 

print(classification_report(y_test, predicted))

draw_cm(y_test, predicted)

print('Accuracy on train set: {:.2f}'.format(model_new.score(X_train, y_train)))

print('Accuracy on test set: {:.2f}'.format(model_new.score(X_test, y_test)))

print('Recall score: {:.2f}'.format(recall_score(y_test,predicted)))

print('ROC AUC score: {:.2f}'.format(roc_auc_score(y_test,predicted)))

print('Precision score: {:.2f}'.format(precision_score(y_test,predicted)))



viz = ClassificationReport(model_new)

viz.fit(X_train,y_train)

viz.score(X_test,y_test)

viz.show()