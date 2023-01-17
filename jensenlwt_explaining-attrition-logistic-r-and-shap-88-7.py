# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Standard python import

import math, datetime, os 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation 

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as mn



# Stats

from scipy import stats



# ML

from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier, plot_importance



import lime

import shap

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, ClassPredictionError, PrecisionRecallCurve

from yellowbrick.features import FeatureImportances

from yellowbrick.model_selection import LearningCurve, ValidationCurve





# Setting parameters for plotting 

plt.rcParams['figure.figsize'] = 8,6

plt.rcParams['image.cmap'] = 'viridis'

plt.style.use('ggplot')

%config InlineBackend.figure_format = 'png'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import csv

employee = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

employee.head()
# Encoding attrition to binary variable 

employee['Attrition'] = np.where(employee.Attrition=='Yes',1,0)
# Inspecting the types of variables in the dataset

employee.dtypes
# Retrieving the categorical variables

categorical = employee.select_dtypes(include='object')

print('There are {} categorical variables'.format(len(categorical.columns)))
# Retrieving the numerical variables

numerical = employee.select_dtypes(include=['int64','float64'])

print('There are {} numerical variables'.format(len(numerical.columns)))
# Viewing the categorical variables

categorical.head()
# Viewing the numerical variables 

numerical.head()
# Understanding the values in discrete variables

for var in ["DistanceFromHome", "Education", "EnvironmentSatisfaction", 

            "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", 

            "RelationshipSatisfaction", 'TrainingTimesLastYear']:

    print(var, 'values: ', employee[var].unique())
# Number of missing values

employee.isnull().mean()
non_cont = ['Attrition', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 

            'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

           'DistanceFromHome', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'StockOptionLevel']

continuous = [var for var in numerical.columns if var not in non_cont]

continuous
# Let's create boxplot to visualise the outliers in the continous variables

for var in continuous:

    plt.figure(figsize=(10,4), dpi=300)

    plt.subplot(1,2,1)

    fig = employee.boxplot(column=var)

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1,2,2)

    fig = employee[var].hist(bins=20)

    fig.set_ylabel('Number of employees')

    fig.set_xlabel(var)

    

    plt.show()
# Outliers in discrete variables

discrete = []

for var in employee.columns:

    if len(employee[var].unique()) <20:

        discrete.append(var)

        

discrete = [var for var in discrete if var not in ['StandardHours', 'EmployeeCount', 'StockOptionLevel', 'EmployeeCount', 'EmployeeNumber']]

discrete
for var in discrete:

    print(employee[var].value_counts()/np.float(len(employee)))

    print()
for var in categorical.columns:

    print(var, 'contains', len(employee[var].unique()), 'labels')
# BusinessTravel

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(x='Attrition', data=employee, palette='viridis',

                     saturation=1,ax=ax)
# Gender

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='Gender', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# BusinessTravel

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='BusinessTravel', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# Department

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='Department', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# EducationField

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='EducationField', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# Gender

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='Gender', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
 # JobRole

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(25,6), dpi=300)

    _ = sns.countplot(x='JobRole', data=employee, palette='viridis',

                     saturation=1,ax=ax)
# MaritalStatus

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='MaritalStatus', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# OverTime

with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    _ = sns.countplot(y='OverTime', data=employee, hue='Attrition', palette='viridis',

                     saturation=1,ax=ax)
# Drop features with constant values and redundant features

employee = employee.drop(['StandardHours','Over18','EmployeeCount', 'EmployeeNumber'], axis=1)
# Checking dataframe

employee.head()
# Seperating into train and test set



X_train, X_test, y_train, y_test = train_test_split(employee, employee.Attrition, test_size=0.2, random_state=0)

X_train.shape, X_test.shape
# Check shape

employee.shape
def tree_binariser(var):

    score_ls = []



    for tree_depth in [1,2,3,4]:

        # Calling the model

        tree_model = DecisionTreeRegressor(max_depth=tree_depth)



        # Train the model with 3 fold CV

        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')

        score_ls.append(np.mean(scores))



    # Finding the depth with the smallest MSE

    depth = [1,2,3,4][np.argmax(score_ls)]

    #print(score_ls, np.argmax(score_ls), depth)



    # Transform the continous variable with the tree

    tree_model = DecisionTreeRegressor(max_depth=depth)

    tree_model.fit(X_train[var].to_frame(), X_train.Attrition)

    X_train[var] = tree_model.predict(X_train[var].to_frame())

    X_test[var] = tree_model.predict(X_test[var].to_frame())
# Transform the continuous variables

for var in continuous:

    tree_binariser(var)
X_train[continuous].head()
# Check the number of bins in each continuous variables

for var in continuous:

    print(var, len(X_train[var].unique()))
# Initialising LabelEncoder()

le = LabelEncoder()



# Retrieving categorical columns

categorical = employee.select_dtypes(include='object')

categorical = categorical.columns



for var in categorical:

    X_train[var] = le.fit_transform(X_train[var])

    X_test[var] = le.fit_transform(X_test[var])
X_train.head()
# Creating dummy variables for all categorical features



cat = ["DistanceFromHome", "Education", "EnvironmentSatisfaction", 

            "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", 

            "RelationshipSatisfaction", 'TrainingTimesLastYear', "BusinessTravel",

        "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime", 'WorkLifeBalance',

      'StockOptionLevel', 'NumCompaniesWorked']



for var in cat:

    X_train[var] = X_train[var].astype('object')

    X_test[var] = X_test[var].astype('object')

    

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
X_train.head()
# Drop attrition

X_train = X_train.drop('Attrition', axis=1)

X_test = X_test.drop('Attrition', axis=1)
# Initialise StandardScaler

sc = StandardScaler()

sc.fit(X_train)
# 1st model - Logistic Regression 

logr = LogisticRegression()

logr.fit(sc.transform(X_train), y_train)

logr.score(sc.transform(X_test), y_test), cross_val_score(logr, sc.transform(X_test), y_test, cv=5).mean()
# Plotting confusion matrix for logr

with sns.plotting_context('paper'):

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)

    cm_viz = ConfusionMatrix(logr, cmap=False, percent=False)

    cm_viz.fit(sc.transform(X_train), y_train)

    cm_viz.score(sc.transform(X_test), y_test)

    cm_viz.poof()
# Classification report for logr

print(classification_report(y_test, logr.predict(sc.transform(X_test))))
# 2nd model - Random Forest

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf.score(X_test, y_test), cross_val_score(rf, X_test, y_test, cv=5).mean()
# Plotting confusion matrix for rf

with sns.plotting_context('paper'):

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)

    cm_viz = ConfusionMatrix(rf, cmap=False, percent=False)

    cm_viz.fit(X_train, y_train)

    cm_viz.score(X_test, y_test)

    cm_viz.poof()
# Classification report for logr

print(classification_report(y_test, rf.predict(X_test)))
xgb = XGBClassifier()



xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb.score(X_test, y_test), cross_val_score(xgb, X_test, y_test, cv=5).mean()
# Plotting confusion matrix for xgb

with sns.plotting_context('paper'):

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)

    cm_viz = ConfusionMatrix(xgb, cmap=False, percent=False)

    cm_viz.fit(X_train, y_train)

    cm_viz.score(X_test, y_test)

    cm_viz.poof()
# Classification report for logr

print(classification_report(y_test, xgb.predict(X_test)))
# Plotting ROC curve for logr, rf, xgb



with sns.plotting_context('notebook'):

    fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,8), dpi=300)

    roc_viz = ROCAUC(logr, ax=ax, micro=False)

    roc_viz.score(X_test, y_test)

    roc_viz.finalize()

    roc_viz2 = ROCAUC(rf, ax=ax2, micro=False)

    roc_viz2.score(X_test, y_test)

    roc_viz2.finalize()

    roc_viz3 = ROCAUC(xgb, ax=ax3, micro=False)

    roc_viz3.score(X_test, y_test)

    roc_viz3.finalize()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=45)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())
logr = LogisticRegression()

logr.fit(sc.transform(X_train_new), y_train_new)

logr.score(sc.transform(X_test), y_test), cross_val_score(logr, sc.transform(X_test), y_test, cv=5).mean()
print(classification_report(y_test, logr.predict(sc.transform(X_test))))
# 2nd model - Random Forest

rf = RandomForestClassifier()

rf.fit(X_train_new, y_train_new)

rf.score(X_test, y_test), cross_val_score(rf, X_test, y_test, cv=5).mean()
print(classification_report(y_test, rf.predict(X_test)))
xgb = XGBClassifier()



xgb.fit(X_train_new, y_train_new, eval_set=[(X_test, y_test)], verbose=False)

xgb.score(X_test, y_test), cross_val_score(xgb, X_test, y_test, cv=5).mean()
print(classification_report(y_test, xgb.predict(X_test)))
%%time

# Hyperparameter turning of logr

param_grid = {

    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],

    'penalty' : ['l1', 'l2'],

    'C' : [100, 10, 1.0, 0.1, 0.01]

}



# Instantiate the grid search

logr_g = GridSearchCV(logr, param_grid=param_grid, n_jobs=-1, verbose=0, cv=5, error_score=0)

logr_g.fit(sc.transform(X_train), y_train)

# Summarizing results

print("Best: %f using %s" % (logr_g.best_score_, logr_g.get_params()))

print("\n")
logr_g.score(sc.transform(X_test), y_test), cross_val_score(logr_g, sc.transform(X_test), y_test, cv=5).mean()
logr_g = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
# Using untuned model's parameter

logr_g.fit(sc.transform(X_train_new), y_train_new)

logr_g.score(sc.transform(X_test), y_test), cross_val_score(logr_g, sc.transform(X_test), y_test, cv=5).mean()
print(classification_report(y_test, logr_g.predict(sc.transform(X_test))))
with sns.plotting_context('paper'):

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)

    cm_viz = ConfusionMatrix(logr_g, cmap=False, percent=False)

    cm_viz.fit(sc.transform(X_train), y_train)

    cm_viz.score(sc.transform(X_test), y_test)

    cm_viz.poof()
# Initialising js

shap.initjs()



# Create a tree explainer and understanding the values we have 

shap_ex = shap.LinearExplainer(logr_g, X_test)

vals = shap_ex.shap_values(X_test)
# Looking at feature importance 

shap.summary_plot(vals, X_test, plot_type="bar")
# Plotting a summary plot to see how the value of the features help us in predicting the patients



with sns.plotting_context('talk'):

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    shap.summary_plot(vals, X_test, alpha=.5)
# Creating a force plot to explain the first 100 samples

shap.force_plot(shap_ex.expected_value, vals[:100], X_test.iloc[:100])
# Retrieving employee's 50 details

X_test.iloc[[50]]
# Predicting using the logr_g

logr_g.predict(sc.transform(X_test.iloc[[50]]))
# Explaining why no.49 is classified as no employee attrition.

shap.force_plot(shap_ex.expected_value, vals[50,:], X_test.iloc[50,:])
# Convert dataframe to a matrix 

logr_g.fit(X_test.as_matrix(), y_test.as_matrix())



explainer = lime.lime_tabular.LimeTabularExplainer(

    X_test.values,

    feature_names=X_test.columns,

    class_names=[0,1]

)



# Taking row 50 and intepreting the prediction

pos = 50

exp = explainer.explain_instance(X_test.iloc[pos].values, 

                                 logr_g.predict_proba)

_ = exp.show_in_notebook()