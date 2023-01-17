# This Python 2.7.14 environment on ANACONDA 4.5.0 with pip 9.0.3; running in ubuntu LTS with a API_key to kaggle
#or running in iphyton trough jupyter notebook

#misc libraries
import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

import sys #access to system parameters https://docs.python.org/3/library/sys.html
import os #not sure what is this (maybe for python in linux kernel)
print("Python version: {}". format(sys.version))
path=os.getcwd()
print(path)

#load packages
import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 
import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))
import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

#visualization
import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import matplotlib as mpl #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(mpl.__version__))
import seaborn as sns
print("Seaborn version: {}". format(sns.__version__))

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))
import xgboost as xgb #collection ???
print("Xgboost version: {}". format(xgb.__version__))


# Input data files are available in the /home/ghillaz/.kaggle/competitions/titanic
#Import Data sets and read
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

#to play with our data we'll create a copy
#remember python assignment or equal passes by reference vs values, so we use the copy function from pandas
train1 = train.copy(deep = True)

#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [train1, test] #is a list (data_cleaner[0] to access)

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
print(train.info())
print(train.sample(3))
#################################################
print("#######################")

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection, model_selection, metrics

#Visualization
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
print("**START**")
print("#######################")

#In this stage, we will clean our data by 1) correcting aberrant values and outliers,
#2) completing missing information, 3) creating new features for analysis, and
#4) converting fields to the correct format for calculations and presentation.

print('Train columns with null values:\n', train1.isnull().sum())#show how many null values have for column
print("-"*10)

print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)

train.describe(include = 'all')#show the data info descreption

#Clean Data
# pandas.DataFrame ; pandas.DataFrame.info ; pandas.DataFrame.describe
# Indexing and Selecting Data
# pandas.isnull ; pandas.DataFrame.sum ;pandas.DataFrame.mode ; pandas.DataFrame.copy ;
# pandas.DataFrame.fillna ; pandas.DataFrame.drop ; pandas.Series.value_counts ; pandas.DataFrame.loc

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
train1.drop(drop_column, axis=1, inplace = True)

print(train1.isnull().sum())
print("-"*10)
print(test.isnull().sum())

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
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 6) 
    # Age Buckets (6 buckets): 0-13.33;13.33-26.67...


    
#cleanup rare title names
#print(train1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (train1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
train1['Title'] = train1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(train1['Title'].value_counts())
print("-"*10)

#preview data again
train1.info()
test.info()
train1.sample(10)

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
train1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
train1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
train1_xy =  Target + train1_x
print('Original X Y: ', train1_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
train1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
train1_xy_bin = Target + train1_x_bin
print('Bin X Y: ', train1_xy_bin, '\n')


#define x and y variables for dummy features original
train1_dummy = pd.get_dummies(train1[train1_x])
train1_x_dummy = train1_dummy.columns.tolist()
train1_xy_dummy = Target + train1_x_dummy
print('Dummy X Y: ', train1_xy_dummy, '\n')

train1_dummy.head()

################Da-Double Check Cleaned Data

print('Train columns with null values: \n', train1.isnull().sum())
print("-"*10)
print (train1.info())
print("-"*10)

print('Test/Validation columns with null values: \n', test.isnull().sum())
print("-"*10)
print (test.info())
print("-"*10)

train.describe(include = 'all')

#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train1[train1_x_calc], train1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train1[train1_x_bin], train1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(train1_dummy[train1_x_dummy], train1[Target], random_state = 0)


print("train1 Shape: {}".format(train1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()

#Discrete Variable Correlation by Survival using
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in train1_x:
    if train1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(train1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        

#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
print(pd.crosstab(train1['Title'],train1[Target[0]]))

#IMPORTANT: Intentionally plotted different ways for learning purposes only. 

#optional plotting w/pandas: https://pandas.pydata.org/pandas-docs/stable/visualization.html

#we will use matplotlib.pyplot: https://matplotlib.org/api/pyplot_api.html

#to organize our graphics will use figure: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=train1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(train1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(train1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [train1[train1['Survived']==1]['Fare'], train1[train1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [train1[train1['Survived']==1]['Age'], train1[train1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [train1[train1['Survived']==1]['FamilySize'], train1[train1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
