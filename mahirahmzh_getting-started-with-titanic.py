import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#load packages

import sys

import pandas as pd

import matplotlib

import numpy as np

import scipy as sp

import IPython

from IPython import display

import sklearn

import random

import time

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neighbors

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



#Configure Visualization Defaults

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_raw = pd.read_csv('../input/titanic/train.csv')

data_val  = pd.read_csv('../input/titanic/test.csv')





#to play with our data we'll create a copy

#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs

data1 = data_raw.copy(deep = True)



#however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data1, data_val]





#preview data

print (data_raw.info()) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html

#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html

#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html

data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', data_val.isnull().sum())

print("-"*10)



data_raw.describe(include = 'all')
women = data_raw.loc[data_raw.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = data_raw.loc[data_raw.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
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

label = LabelEncoder()

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
#split train and test data with function defaults

#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)





print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))



train1_x_bin.head()
#Discrete Variable Correlation by Survival using

#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

for x in data1_x:

    if data1[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')

        



#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html

print(pd.crosstab(data1['Title'],data1[Target[0]]))
#graph distribution of quantitative data

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

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
#Machine Learning Algorithm (MLA) Selection and Initialization

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





#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

#note: this is an alternative to train_test_split

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#create table to compare MLA predictions

MLA_predict = data1[Target]



#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv  = cv_split)

    



    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

   # MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(data1[data1_x_bin], data1[Target])

    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    

    row_index+=1



    

#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare

#MLA_predict
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare)



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
#group by or pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

pivot_female = data1[data1.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','FareBin'])['Survived'].mean()

print('Survival Decision Tree w/Female Node: \n',pivot_female)



pivot_male = data1[data1.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)
#handmade data model using brain power (and Microsoft Excel Pivot Tables for quick calculations)

def mytree(df):

    

    #initialize table to store predictions

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = ['Master'] #survived titles



    for index, row in df.iterrows():



        #Question 1: Were you on the Titanic; majority died

        Model.loc[index, 'Predict'] = 0



        #Question 2: Are you female; majority survived

        if (df.loc[index, 'Sex'] == 'female'):

                  Model.loc[index, 'Predict'] = 1



        #Question 3A Female - Class and Question 4 Embarked gain minimum information



        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       

        if ((df.loc[index, 'Sex'] == 'female') & 

            (df.loc[index, 'Pclass'] == 3) & 

            (df.loc[index, 'Embarked'] == 'S')  &

            (df.loc[index, 'Fare'] > 8)



           ):

                  Model.loc[index, 'Predict'] = 0



        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

        if ((df.loc[index, 'Sex'] == 'male') &

            (df.loc[index, 'Title'] in male_title)

            ):

            Model.loc[index, 'Predict'] = 1

        

        

    return Model





#model data

Tree_Predict = mytree(data1)

print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))





#Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report

#Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score

#And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

print(metrics.classification_report(data1['Survived'], Tree_Predict))
#Plot Accuracy Summary

#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Compute confusion matrix

cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict)

np.set_printoptions(precision=2)



class_names = ['Dead', 'Survived']

# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')
#base model

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv  = cv_split)

dtree.fit(data1[data1_x_bin], data1[Target])



print('BEFORE DT Parameters: ', dtree.get_params())

#print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))

print('-'*10)





#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best

              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

             }



#print(list(model_selection.ParameterGrid(param_grid)))



#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

tune_model.fit(data1[data1_x_bin], data1[Target])



#print(tune_model.cv_results_.keys())

#print(tune_model.cv_results_['params'])

print('AFTER DT Parameters: ', tune_model.best_params_)

#print(tune_model.cv_results_['mean_train_score'])

#print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(tune_model.cv_results_['mean_test_score'])

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
#base model

print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape) 

print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)



#print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)







#feature selection

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(data1[data1_x_bin], data1[Target])



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv  = cv_split)



#print(dtree_rfe.grid_scores_)

print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



#print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)





#tune rfe model

rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

rfe_tune_model.fit(data1[X_rfe], data1[Target])



#print(rfe_tune_model.cv_results_.keys())

#print(rfe_tune_model.cv_results_['params'])

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

#print(rfe_tune_model.cv_results_['mean_train_score'])

#print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(rfe_tune_model.cv_results_['mean_test_score'])

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
#why choose one model, when you can pick them all with voting classifier

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

#removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model

vote_est = [

    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

    ('ada', ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

    ('etc',ensemble.ExtraTreesClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),



    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc

    ('gpc', gaussian_process.GaussianProcessClassifier()),

    

    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ('lr', linear_model.LogisticRegressionCV()),

    

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html

    ('bnb', naive_bayes.BernoulliNB()),

    ('gnb', naive_bayes.GaussianNB()),

    

    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html

    ('knn', neighbors.KNeighborsClassifier()),

    

    #SVM: http://scikit-learn.org/stable/modules/svm.html

    ('svc', svm.SVC(probability=True)),

    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

   ('xgb', XGBClassifier())



]



#Hard Vote or majority rules

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv  = cv_split)

vote_hard.fit(data1[data1_x_bin], data1[Target])



#print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)





#Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)

vote_soft.fit(data1[data1_x_bin], data1[Target])



#print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 

print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)
#Hard Vote or majority rules w/Tuned Hyperparameters

grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv  = cv_split)

grid_hard.fit(data1[data1_x_bin], data1[Target])



#print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)



#Soft Vote or weighted probabilities w/Tuned Hyperparameters

grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)

grid_soft.fit(data1[data1_x_bin], data1[Target])



#print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 

print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)
print(data_val.info())

print("-"*10)

#data_val.sample(10)







#handmade decision tree - submission score = 0.77990

data_val['Survived'] = mytree(data_val).astype(int)





#decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990

#submit_dt = tree.DecisionTreeClassifier()

#submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

#submit_dt.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}

#data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])





#bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990

#submit_bc = ensemble.BaggingClassifier()

#submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_bc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_bc.best_params_) #Best Parameters:  {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}

#data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])





#extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990

#submit_etc = ensemble.ExtraTreesClassifier()

#submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_etc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_etc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}

#data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])





#random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205

#submit_rfc = ensemble.RandomForestClassifier()

#submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_rfc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}

#data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])







#ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119

#submit_abc = ensemble.AdaBoostClassifier()

#submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_abc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_abc.best_params_) #Best Parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}

#data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])





#gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033

#submit_gbc = ensemble.GradientBoostingClassifier()

#submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_gbc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters:  {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}

#data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])



#extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990

#submit_xgb = XGBClassifier()

#submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_xgb.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}

#data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])





#hard voting classifier w/full dataset modeling submission score: defaults= 0.75598, tuned = 0.77990

#data_val['Survived'] = vote_hard.predict(data_val[data1_x_bin])

data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])





#soft voting classifier w/full dataset modeling submission score: defaults= 0.73684, tuned = 0.74162

#data_val['Survived'] = vote_soft.predict(data_val[data1_x_bin])

#data_val['Survived'] = grid_soft.predict(data_val[data1_x_bin])





#submit file

submit = data_val[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)



print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True))

submit.sample(10)