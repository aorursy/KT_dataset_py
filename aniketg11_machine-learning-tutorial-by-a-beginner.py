# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib #collection of functions for scientific and publication-ready visualization
import scipy as sp #collection of functions for scientific computing and advance mathematics
import IPython #pretty printing of dataframes in Jupyter notebook
from IPython import display
import sklearn #collection of machine learning algorithms

#misc libraries
import random
import time

#ignore warning
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load Modelling Algorithms
#We will use the popular scikit-learn library to develop our machine learning algorithms. 
#In sklearn, algorithms are called Estimators and implemented in their own classes. For data visualization, we will use the matplotlib and seaborn library. Below are common classes to load.
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


#visualization defaults
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8

#lets first import our data
data_raw = pd.read_csv('../input/train.csv')
data_val  = pd.read_csv('../input/test.csv')

#Make a deep copy, including a copy of the data and the indices. With deep=False neither the indices nor the data are copied.
data1 = data_raw.copy(deep= True)
#in order to clean both datasets at once
data_cleaner = [data1, data_val]

#preview data

data_raw.info()

data_raw.sample(10)

print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())
data1.describe(include='all')
#COMPLETE or delete missing values in train and test/validation dataset
for dataset in data_cleaner:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace= True)
    #complete missing Embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the PassengerId, Cabin and Ticket feature to exclude in train dataset
drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)
    
print('Train columns dropped')

    
    


print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())
drop_column = ['Cabin', 'Ticket']
data_val.drop(drop_column, axis = 1, inplace=True)
print('Test columns with null values:\n', data_val.isnull().sum())
###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    #If other family members are present IsAlone will be 0
    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0
    #extracting title from data
    dataset['Title'] = dataset['Name'].str.split(', ', expand = True)[1].str.split(".", expand = True)[0]
     
    #With qcut, the bins will be chosen so that you have the same number of records in each bin 
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    
    #cut will choose the bins to be evenly spaced according to the values themselves and not the frequency of those values

    
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
print('Feature Engineering done for FareBin and AgeBin')
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
print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())
data_raw.describe(include = 'all')
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()
for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Survival correlation by ', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())

        
        
print(pd.crosstab(data1['Title'], data1[Target[0]]))

#graph distribution of quantitative data
plt.figure(figsize=[16, 12])
plt.subplot(231)
plt.boxplot(x = data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x = data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.boxplot(x = data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Familysize')
plt.ylabel('Familysize (Nos)')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']],stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare histogram by survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], stacked = True,color= ['g','b'], label = ['Survived', 'Dead'])
plt.title('Age histogram by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], stacked = True, color = ['r','g'], label = ['Survived', 'Dead'], bins = 10)
plt.title('Family size histogram by survival')
plt.xlabel('Family Size')
plt.ylabel('# of Passengers')
plt.legend()

fig, saxis = plt.subplots(2, 3, figsize = (16, 12))
sns.barplot(x= 'Embarked', y='Survived', data= data1, ax= saxis[0,0])
sns.barplot(x='Pclass', y='Survived', data=data1, ax=saxis[0,1])
sns.barplot(x='IsAlone', y='Survived', data= data1, ax= saxis[0,2])

sns.pointplot(x='FareBin', y='Survived', data=data1, ax= saxis[1,0])
sns.pointplot(x='AgeBin', y='Survived', data=data1, ax= saxis[1,1])
sns.pointplot(x='FamilySize', y='Survived', data=data1, ax= saxis[1,2])




#we know sex mattered in survival, now let's compare sex and Embarked

fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax = qaxis[1])
sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax = qaxis[2])
            


fig, qaxis = plt.subplots(1, 3, figsize = (14,12))
sns.distplot(data1["Age"],kde=True, ax =qaxis[0]) #without the kde
sns.distplot(data1['Fare'], kde = True, ax=qaxis[1])
sns.distplot(data1['FamilySize'], kde = True, ax=qaxis[2])

def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(16, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(df.corr(), annot= True, cmap=colormap)
    
        
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
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest neighbour
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability= True),
    svm.LinearSVC(),
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    XGBClassifier()

]

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
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
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
sns.barplot(x= 'MLA Test Accuracy Mean', y='MLA Name', data = MLA_compare, color='m')
plt.title('MLA Accuracy score')
plt.xlabel('Accuracy score %')
plt.ylabel('Algorithm name')

clf = svm.SVC()
clf.fit(data1[data1_x_bin], data1[Target])
pred= clf.predict(data_val[data1_x_bin])
submission = pd.DataFrame({"PassengerId": data_val['PassengerId'], "Survived": pred})
#data_val.columns
submission.to_csv("../working/submission.csv", index=False)

submission.sample(10)