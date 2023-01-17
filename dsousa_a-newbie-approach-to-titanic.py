import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Advance mathematics

import scipy as sp



# Pretty print dataframes

import IPython

from IPython import display



# Misc

import random

import time
# Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



# Model Utilities

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Data Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



# Data Visualization Backend

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_train = pd.read_csv('../input/train.csv')

df_val  = pd.read_csv('../input/test.csv')



# Copy and use them in an list to clean both at the same time

df_train = data_train.copy(deep = True)

dfs = [df_train, df_val]



# Preview of data

df_train.head()
print('Train columns with null values:\n', df_train.isnull().sum())

print('--' * 15)

print('--' * 15)



print('Test columns with null values:\n', df_val.isnull().sum())

print('--' * 15)

print('--' * 15)
# We know what we need to clean (or fill) so we do it in both dataframes

for dataset in dfs:

    

    # Complete numerical missing values with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    # Complete alphabetical missing values with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



# Drop this columns because PassengerId and Ticket does not gives relevant info

# about the passenger.

# The Cabin could be usefull to know the position of the passenger cabin

# in the boat but it has too much missing values so we drop it in the train data.

drop_column = ['PassengerId','Cabin', 'Ticket']

df_train.drop(drop_column, axis=1, inplace = True)
# Check again the missing values in the datasets

print('Train columns with null values:\n', df_train.isnull().sum())

print('--' * 15)

print('--' * 15)



print('Test columns with null values:\n', df_val.isnull().sum())

print('--' * 15)

print('--' * 15)
# Let's try to extract information to create new features

for dataset in dfs:

    

    # Family size may be relevant

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    

    # Passenger is alone or not, depending on the family size

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    

    # Cut the Fare and the Age in intervals to better understanding

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    

    # Get the title of the passenger

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



# Using the common minimial for stats

stat_min = 10



# Ger rid of strange Titles (count below the minimum)

title_names = (df_train['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

df_train['Title'] = df_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



# Check the data

df_train.info()

df_train.head()
label = LabelEncoder()



for dataset in dfs:

    

    # Convert to dummy variables those columns with few different values

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])



input_columns = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

target = ['Survived']
MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]



# Run the models 10x times with train/val of 60/30 leaving out the 10%

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )



# Table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



# Table to compare MLA predictions vs the expected output

MLA_predict = df_train[target]



row_index = 0

for alg in MLA:



    # Name of the MLA

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    

    # Cross Validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

    cv_results = model_selection.cross_validate(alg, df_train[input_columns], df_train[target], cv  = cv_split, return_train_score = True)



    # Store the results in the table

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    

    # Save the prediction in the corresponding table

    alg.fit(df_train[input_columns], df_train[target])

    MLA_predict[MLA_name] = alg.predict(df_train[input_columns])

    

    row_index+=1



# Sort results by descending Test Score

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
# Base Tree

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, df_train[input_columns],df_train[target], cv  = cv_split, return_train_score = True)

dtree.fit(df_train[input_columns], df_train[target])



print('BEFORE DT Parameters: ', dtree.get_params())

print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)



# Parameters to search best estimator like criterion gini (by default),max depth of the tree and random seed

param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8,10,None], 'random_state': [0]}

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score = True)



# Fit the data in the new estimator

tune_model.fit(df_train[input_columns], df_train[target])



print('AFTER DT Parameters: ', tune_model.best_params_)

print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# Base Data & Model Score

print('BEFORE DT RFE Training Shape Old: ', df_train[input_columns].shape) 

print('BEFORE DT RFE Training Columns Old: ', df_train[input_columns].columns.values)

print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)



# Feature Selection

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy')

dtree_rfe.fit(df_train[input_columns], df_train[target])



# Reduce features with the most relevant columns

X_rfe = df_train[input_columns].columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, df_train[X_rfe], df_train[target], cv  = cv_split, return_train_score = True)



# Data & Model Score after feature selection with base model

print('AFTER DT RFE Training Shape New: ', df_train[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)

print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)



# Tune model with the new features

rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score = True)

rfe_tune_model.fit(df_train[X_rfe], df_train[target])



# Model Hyper-Params and scores after Tunning

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(MLA_predict)
df_val['Survived'] = rfe_tune_model.predict(df_val[X_rfe])



# Submit file

submission = df_val[['PassengerId','Survived']]

submission.to_csv('submit.csv', index=False)

submission.head()