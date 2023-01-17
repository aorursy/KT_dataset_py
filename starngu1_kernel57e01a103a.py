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



# Any results you write to the current directory are saved as output.
import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))



#misc libraries

import random

import time





#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

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

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_raw = pd.read_csv('../input/titanic/train.csv')

data_val  = pd.read_csv('../input/titanic/test.csv')

data1 = data_raw.copy(deep = True)

data_cleaner = [data1, data_val]

print (data_raw.info())

data_raw.sample(10)
print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', data_val.isnull().sum())

print("-"*10)



data_raw.describe(include = 'all')
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
print('Train columns with null values: \n', data1.isnull().sum())

print("-"*10)

print (data1.info())

print("-"*10)



print('Test/Validation columns with null values: \n', data_val.isnull().sum())

print("-"*10)

print (data_val.info())

print("-"*10)



data_raw.describe(include = 'all')
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)





print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))



train1_x_bin.head()
for x in data1_x:

    if data1[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')

        



#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html

print(pd.crosstab(data1['Title'],data1[Target[0]]))
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
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
fig, qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does family size factor with sex & survival compare

sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)



#how does class factor with sex & survival compare

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
#how does embark port factor with class, sex, and survival compare

#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

e = sns.FacetGrid(data1, col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
#plot distributions of age of passengers who survived or did not survive

a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , data1['Age'].max()))

a.add_legend()
#histogram comparison of sex, class, and age by survival

h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
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

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(data1[data1_x_bin], data1[Target])

    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    

    row_index+=1



    

#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
pivot_female = data1[data1.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','FareBin'])['Survived'].mean()

print('Survival Decision Tree w/Female Node: \n',pivot_female)



pivot_male = data1[data1.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)
dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv  = cv_split)

dtree.fit(data1[data1_x_bin], data1[Target])



print('BEFORE DT Parameters: ', dtree.get_params()) 

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

#print(tune_model.cv_results_['mean_test_score'])

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
#base model

print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape) 

print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values) 

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

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
import graphviz 

dot_data = tree.export_graphviz(dtree, out_file=None, 

                                feature_names = data1_x_bin, class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph
#compare algorithm predictions with each other, where 1 = exactly similar and 0 = exactly opposite

#there are some 1's, but enough blues and light reds to create a "super algorithm" by combining them

correlation_heatmap(MLA_predict)
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



print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)





#Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)

vote_soft.fit(data1[data1_x_bin], data1[Target])



print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)
#WARNING: Running is very computational intensive and time expensive.

#Code is written for experimental/developmental purposes and not production ready!





#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]





grid_param = [

            [{

            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

            'n_estimators': grid_n_estimator, #default=50

            'learning_rate': grid_learn, #default=1

            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R

            'random_state': grid_seed

            }],

       

    

            [{

            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

            'n_estimators': grid_n_estimator, #default=10

            'max_samples': grid_ratio, #default=1.0

            'random_state': grid_seed

             }],



    

            [{

            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'random_state': grid_seed

             }],





            [{

            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

            #'loss': ['deviance', 'exponential'], #default=’deviance’

            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”

            'max_depth': grid_max_depth, #default=3   

            'random_state': grid_seed

             }],



    

            [{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

            'random_state': grid_seed

             }],

    

            [{    

            #GaussianProcessClassifier

            'max_iter_predict': grid_n_estimator, #default: 100

            'random_state': grid_seed

            }],

        

    

            [{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': grid_bool, #default: True

            #'penalty': ['l1','l2'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

            'random_state': grid_seed

             }],

            

    

            [{

            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

            'alpha': grid_ratio, #default: 1.0

             }],

    

    

            #GaussianNB - 

            [{}],

    

            [{

            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

            'weights': ['uniform', 'distance'], #default = ‘uniform’

            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }],

            

    

            [{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            'C': [1,2,3,4,5], #default=1.0

            'gamma': grid_ratio, #edfault: auto

            'decision_function_shape': ['ovo', 'ovr'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }],



    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }]   

        ]







start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip



    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

    #print(param)

    

    

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(data1[data1_x_bin], data1[Target])

    run = time.perf_counter() - start



    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) 





run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)
#Hard Vote or majority rules w/Tuned Hyperparameters

grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv  = cv_split)

grid_hard.fit(data1[data1_x_bin], data1[Target])

 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)



#Soft Vote or weighted probabilities w/Tuned Hyperparameters

grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)

grid_soft.fit(data1[data1_x_bin], data1[Target])



print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)
#prepare data for modeling

print(data_val.info())

print("-"*10)

#data_val.sample(10)



 



#handmade decision tree - submission score = 0.77990

#data_val['Survived'] = mytree(data_val).astype(int)





#decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990

submit_dt = tree.DecisionTreeClassifier()

submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

submit_dt.fit(data1[data1_x_bin], data1[Target])

print('Best Parameters: ', submit_dt.best_params_) #Best Parameters: {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}

data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])





#bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990

#submit_bc = ensemble.BaggingClassifier()

#submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_bc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_bc.best_params_) #Best Parameters: {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}

#data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])





#extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990

#submit_etc = ensemble.ExtraTreesClassifier()

#submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_etc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_etc.best_params_) #Best Parameters: {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}

#data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])





#random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205

#submit_rfc = ensemble.RandomForestClassifier()

#submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_rfc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters: {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}

#data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])



 



#ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119

#submit_abc = ensemble.AdaBoostClassifier()

#submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_abc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_abc.best_params_) #Best Parameters: {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}

#data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])





#gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033

#submit_gbc = ensemble.GradientBoostingClassifier()

#submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_gbc.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters: {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}

#data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])



#extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990

#submit_xgb = XGBClassifier()

#submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)

#submit_xgb.fit(data1[data1_x_bin], data1[Target])

#print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters: {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}

#data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])





#hard voting classifier w/full dataset modeling submission score: defaults= 0.75598, tuned = 0.77990

#data_val['Survived'] = vote_hard.predict(data_val[data1_x_bin])

data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])





#soft voting classifier w/full dataset modeling submission score: defaults= 0.73684, tuned = 0.74162

#data_val['Survived'] = vote_soft.predict(data_val[data1_x_bin])

#data_val['Survived'] = grid_soft.predict(data_val[data1_x_bin])





#submit file

submit = data_val[['PassengerId','Survived']]

submit.to_csv("submit.csv", index=False)



print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True))

submit.sample(10)