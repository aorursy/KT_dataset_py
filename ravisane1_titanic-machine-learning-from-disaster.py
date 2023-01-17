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
#load packages

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

#Common Model Algorithms

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
#import data

data_raw = pd.read_csv('/kaggle/input/titanic/train.csv')



#a dataset should be broken into 3 splits: train, test, and (final) validation

#the test file provided is the validation file for competition submission

#we will split the train set into train and test data in future sections

data_val  = pd.read_csv('/kaggle/input/titanic/test.csv')





#to play with our data we'll create a copy

data1 = data_raw.copy(deep = True)



#however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data1, data_val]





#preview data

print (data_raw.info())

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
#split train and test data with function defaults

#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)





print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))



train1_x_bin.head()
#Machine Learning Algorithm (MLA) Selection and Initialization

classifier = ensemble.RandomForestClassifier()



classifier.fit(train1_x_bin, train1_y_bin)



# ## Predicting the Test set results

y_pred = classifier.predict(test1_x_bin)







# ## Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(test1_y_bin, y_pred)

print(cm)

print(accuracy_score(test1_y_bin, y_pred))
data_val['Survived'] = classifier.predict(data_val[data1_x_bin])

submit = data_val[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)