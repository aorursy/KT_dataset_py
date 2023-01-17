# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#print("Hello")

# Any results you write to the current directory are saved as output.

data_raw = pd.read_csv('../input/train.csv')

data_val  = pd.read_csv('../input/test.csv')

#data_val  = pd.read_csv('../input/test.csv')

data1 = data_raw.copy(deep = True)

data_cleaner = [data1, data_val]

print (data_raw.info())

data_raw.sample(10)

#train_df.head(30)
data1.isnull().sum()
data_val.isnull().sum()
data_raw.describe(include = 'all')

###COMPLETING: complete or delete missing values in train and test/validation dataset

for dataset in data_cleaner:

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    

    #Complete emparked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    

    #Complete missing fare with median (there is only one missing fare in test_data, no any in training)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#delete passengerId as it does not contribute in survival, delete cabin as it contain maximum null values

#delete ticket as it does not affect suvival.

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

    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0



#print(data_val.head(40))



#spilit title from name column

for dataset in data_cleaner:

    #split just returns series of list , one list for one row cell, also expand=True will

    #makes sub dataframe, which is splited with row from 0 to till all split

    #Below line returns sub dataframe with row  splited...https://pandas.pydata.org/pandas-docs/stable/text.html

    #dataset['Name'].str.split(",", expand=True)

    dataset['Title'] = dataset['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]

    

#print(data1.head(10))
#converting continuous variables into bin why(https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/)

for dataset in data_cleaner:

    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

print(data1.head(10))
#cleanup rare title names

#print(data1['Title'].value_counts())

stat_min = 10#while small is arbitrary, we'll use the common minimum in 

#statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#print(title_names)

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

    

print(data1.head(5))



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
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
model = XGBClassifier()

model.fit(data1[data1_x_bin], data1[Target] )
data_val_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

y_pred = model.predict(data_val[data_val_x_bin])

print(y_pred)
test = pd.DataFrame(columns = ['PassengerId', 'Survived'])

test['PassengerId'] = data_val['PassengerId']

test['Survived'] = y_pred

test.to_csv('test_result.csv', index=False)