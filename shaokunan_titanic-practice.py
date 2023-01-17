# load packages

import sys

import pandas as pd

import matplotlib

import numpy as np

import scipy as sp

import IPython

from IPython import display

import sklearn

from sklearn import *

from sklearn import preprocessing

import random

import time

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output
# import data from the default file "../input/" 

# a dataset should be broken into 3 splits: train, test, and final validation

# the test file provided is the validation file for competition submission 

# and we will split the train set into train and test data in future sections

data_raw = pd.read_csv('../input/train.csv')

data_val = pd.read_csv('../input/test.csv')

# to play with our data we'll create a copy

data1 = data_raw.copy(deep = True)

data_cleaner = [data1, data_val]



# preview data

print(data_raw.info())

data_raw.sample(10)
print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())

print("-"*10)

data_raw.describe(include = 'all')
###COMPLETING: complete or delete missing values in train and test/validation dataset

for dataset in data_cleaner:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

#delete the cabin feature/column and others previously stated to exclude in train dataset

drop_column = ['PassengerId','Cabin', 'Ticket']

data1.drop(drop_column, axis=1, inplace = True)



print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
###CREATE: Feature Engineering for train and test/validation dataset

for dataset in data_cleaner:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    



#cleanup rare title names

#print(data1['Title'].value_counts())

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(data1['Title'].value_counts())

print("-"*10)





#preview data again

data1.info()

data_val.info()

data1.sample(10)
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset



#code categorical data

label = preprocessing.LabelEncoder()

for dataset in data_cleaner:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])





#define y variable aka target/outcome

Target = ['Survived']



#define x variables for original features aka feature selection

data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

data1_xy =  Target + data1_x

print('Original X Y: ', data1_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

data1_xy_bin = Target + data1_x_bin

print('Bin X Y: ', data1_xy_bin, '\n')





#define x and y variables for dummy features original

data1_dummy = pd.get_dummies(data1[data1_x])

data1_x_dummy = data1_dummy.columns.tolist()

data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y: ', data1_xy_dummy, '\n')







data1_dummy.head()
#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Sex', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived', data=data1, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
#correlation heatmap of dataset

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



correlation_heatmap(data1)
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, ensemble

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics
def train_model(model, features, target, fit=False):

    # Split dataset in cross-validation

    # Run model 10x with 60/30 split intentionally leaving out 10%

    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 

    cv_results = model_selection.cross_validate(model, features, target, cv  = cv_split)

    

    if(fit):

        # fit model

        model = model.fit(features, target)

        return model, cv_results

    

    return cv_results
# Train on a tree

# decision_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

# decision_tree = decision_tree.fit(features, target)

# print(decision_tree.score(features, target))

# print(decision_tree.feature_importances_)

decision_tree = tree.DecisionTreeClassifier()

trained_tree, cv_results = train_model(decision_tree, data1[data1_x_bin], data1[Target], fit=True)



Tree_Predict = trained_tree.predict(data1[data1_x_bin])
# Report

print('Decision Tree Model Accuracy/Precision Score on training set: {:.2f}%\n'

      .format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))

print(metrics.classification_report(data1['Survived'], Tree_Predict))



print('-'*10)

print(np.mean(cv_results['test_score']))
# Make prediciton

prediction_dt = decision_tree.predict(data_val[data1_x_bin])

PassengerId =np.array(data_val["PassengerId"]).astype(int)

solution_dt = pd.DataFrame(prediction_dt, PassengerId, columns = ["Survived"])

solution_dt.to_csv("solution_dt.csv", index_label = ["PassengerId"])
# Tune hyper-parameters

param_grid = {'criterion': ['gini', 'entropy'],  # scoring methodology

              'splitter': ['best', 'random'], # splitting methodology

              'max_depth': [3,4,6], # max depth tree can grow

              'min_samples_split': [2, 3,.03], # minimum subset size BEFORE new split (fraction is % of total)

              'min_samples_leaf': [3, 5, 8, .03,.05], # minimum subset size AFTER new split split (fraction is % of total)

              'random_state': [0] #seed or control random number generator

             }



print(len(list(model_selection.ParameterGrid(param_grid))))



cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 



# Choose best model with grid_search:

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), 

                                          param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

tune_model.fit(data1[data1_x_bin], data1[Target])



# Report

print('-'*10)

print('Best Parameters: ', tune_model.best_params_)

print("Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

print("Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# The best-hyper tree we have now

best_hyper_tree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 4, 

                                              min_samples_leaf = 5, min_samples_split = 2, 

                                              splitter = 'best', random_state = 0)



# Report the best-hyper tree

cv_best = train_model(best_hyper_tree, data1[data1_x_bin], data1[Target])

print(np.mean(cv_best['test_score']))
print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape) 

print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)



#feature selection

dtree_rfe = feature_selection.RFECV(best_hyper_tree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(data1[data1_x_bin], data1[Target])



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(best_hyper_tree, data1[X_rfe], data1[Target], cv  = cv_split)



#print(dtree_rfe.grid_scores_)

print('-'*10)

print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)



# tune rfe model

rfe_tune_model = model_selection.GridSearchCV(best_hyper_tree, param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

rfe_tune_model.fit(data1[X_rfe], data1[Target])



print('-'*10)

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
import graphviz 

best_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 6, 

                                              min_samples_leaf = 3, min_samples_split = 2, 

                                              splitter = 'best', random_state = 0)



best_tree.fit(data1[X_rfe], data1[Target])



dot_data = tree.export_graphviz(best_tree, out_file=None, 

                                feature_names = X_rfe, class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph