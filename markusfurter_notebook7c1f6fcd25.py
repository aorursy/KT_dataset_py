"""



ML performance analysis, model fitting and feature selection were performed as suggested by the following Notebook by AD Freeman:

https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy



"""
"""

Import the relevant modules and ML algorithms

"""



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# IMPORT ML algorithms before running

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







print('Setup complete!')
# importing the .csv datasets as train and test

# train = pd.read_csv('../input/titanic/train.csv')

# test = pd.read_csv('../input/titanic/test.csv')



# train.info() 

# remember that Age, Cabin and Embarked columns contain null values

# columns name, ticket might not be useful for analysis



# convert the Sex column to a binary format

# train.loc[train['Sex'] == 'female', "Sex"] = 0

# train.loc[train['Sex'] == 'male', "Sex"] = 1



# train.head()

# train.info() 

# train.describe()

# there are only two missing values in the Embarked columns. 

# based on the distribution (72% boarded in S), 

# we do assume that these values are S as well

# embark_stats = train.groupby('Embarked').count()

# print(embark_stats)
# replace missing values in Embark by S

# embark_nulls = train.loc[train.Embarked.isnull()]

# train.loc[train['Embarked'].isnull(), "Embarked"] = 'S'



# embark_stats = train.groupby('Embarked').count()



# print(embark_stats)

# embark_nulls.head()
# Test whether individual columns are linearly correlated with each other

# Pclass and Fare are well correlated

# Parch and SibSp correlate well

# Pclass correlates fairly well with Survived!



# columns = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']

# sub_train = abs(train.loc[:,columns].corr())

# sns.heatmap(sub_train)
# Missing Age values could be filled in using statistics on the age column. 

# To this end, select columns with present age information and look at stats.

# Potentially, age can vary between passenger classes. therefore group by class and count stats:



# age_train = train.loc[train["Age"].notnull()]

# age_train.Age.groupby(age_train['Pclass']).describe()
# Age data is not distributed equally across passenger classes!!

# age_train.Age.groupby(age_train['Pclass']).median()
# age correlates fairly well with Pclass, so it makes sense to approximate age based on Pclass

# columns = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']

# sub_age_train = abs(age_train.loc[:,columns].corr())

# sns.heatmap(sub_age_train)
# figure out the relationship between age and SibSp/Parch

# SibSp is high for kids, Parch is high for parents

# leave this out for age correction



# sns.scatterplot(x='Age', y='SibSp', data=train)

# sns.scatterplot(x='Age', y='Parch', data=train)

# replace NaN values for the age column with the median age for the respective Pclass

# train.loc[(train['Age'].isnull())&(train['SibSp'] >= 3), 'Age'] = 9  # Kids with siblings

# train.loc[(train['Age'].isnull())&(train['Pclass']==1), 'Age'] = 37  # 1st class passengers

# train.loc[(train['Age'].isnull())&(train['Pclass']==2), 'Age'] = 29  # 2nd class passengers

# train.loc[(train['Age'].isnull())&(train['Pclass']==3), 'Age'] = 24  # 3rd class passengers





# train.info()
# Age data is not distributed equally across family structure.

# print(age_train.Age.groupby(age_train['SibSp']).median())

# print(age_train.Age.groupby(age_train['Parch']).median())



# incorporate the following condition into the above model for age correction:

# train.loc[(train['Age'].isnull())&(train['SibSp'] >= 3), 'Age'] = 9  # Kids with siblings
# create new column family, which describes the family size

# train['family'] = train.SibSp + train.Parch





# extract titles from the Name columns. Keep what's left of "." and right of ", "

# train['title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



# plot the titles by group

# train.groupby('title').count()



# replace special titles with Mr, Mrs and Miss

# male_titles = ['Master','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']

# train.title.replace(to_replace=male_titles, value='Mr', inplace = True)

# train.title.replace(to_replace=['Mme', 'the Countess', 'Lady'], value='Mrs', inplace = True)

# train.title.replace(to_replace=['Ms', 'Mlle'], value='Miss', inplace = True)



# Assign male/female doctors to Mr / Mrs

# train.loc[(train.title == 'Dr') & (train.Sex == 1), 'title']='Mr'

# train.loc[(train.title == 'Dr') & (train.Sex == 0), 'title']='Mrs'

# cathegorize the Age column into PREDIFINED bins

# age_scalars = [0.1, 20.315, 40.21,60.105, 90.0]

# train['age_bin'] = pd.cut(train['Age'],bins=age_scalars)



# cathegorize the Fare column into PREDIFINED bins

# fare_scalars = [-0.001, 8.662 , 26.0, 912.329]

# train['fare_bin'] = pd.cut(train['Fare'],bins=fare_scalars)



# cathegorize the family column into binary format

# train.loc[train['family']>0, 'family']=1

# drop columns not used for analysis

# to_drop = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

# train.drop(labels = to_drop, axis = 1, inplace= True)

# train.head(10)

"""

This section of the code prepares the data from the training dataset for analysis

"""





# importing the .csv dataset as train 

train = pd.read_csv('../input/titanic/train.csv')



# convert the Sex column to a binary format

train.loc[train['Sex'] == 'female', "Sex"] = 0

train.loc[train['Sex'] == 'male', "Sex"] = 1

train['Sex'] = train['Sex'].astype(int)



# replace missing values in Embark by S

embark_nulls = train.loc[train.Embarked.isnull()]

train.loc[train['Embarked'].isnull(), "Embarked"] = 'S'



# replace NaN values for the age column with the median age for the respective Pclass

train.loc[(train['Age'].isnull())&(train['SibSp'] >= 3), 'Age'] = 9  # Kids with siblings

train.loc[(train['Age'].isnull())&(train['Pclass']==1), 'Age'] = 37  # 1st class passengers

train.loc[(train['Age'].isnull())&(train['Pclass']==2), 'Age'] = 29  # 2nd class passengers

train.loc[(train['Age'].isnull())&(train['Pclass']==3), 'Age'] = 24  # 3rd class passengers



# extract titles from the Name columns. Keep what's left of "." and right of ", "

train['title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



# replace special titles with Mr, Mrs and Miss

male_titles = ['Master','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']

train.title.replace(to_replace=male_titles, value='Mr', inplace = True)

train.title.replace(to_replace=['Mme', 'the Countess', 'Lady'], value='Mrs', inplace = True)

train.title.replace(to_replace=['Ms', 'Mlle'], value='Miss', inplace = True)



# Assign male/female doctors to Mr / Mrs

train.loc[(train.title == 'Dr') & (train.Sex == 1), 'title']='Mr'

train.loc[(train.title == 'Dr') & (train.Sex == 0), 'title']='Mrs'



# cathegorize the Age column into PREDIFINED bins

age_scalars = [0.1, 20.315, 40.21,60.105, 90.0]

train['age_bin'] = pd.cut(train['Age'],bins=age_scalars)



# cathegorize the Fare column into PREDIFINED bins

fare_scalars = [-0.001, 8.662 , 26.0, 912.329]

train['fare_bin'] = pd.cut(train['Fare'],bins=fare_scalars)



# create new column family, which describes the family size

train['family'] = train.SibSp + train.Parch

# cathegorize the family column into binary format

train.loc[train['family']>0, 'family']=1



# drop columns not used for analysis

to_drop = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

train.drop(labels = to_drop, axis = 1, inplace= True)



# Generate dummy vectors for the  columns in dummy_list

dummy_list = ['title', 'age_bin', 'fare_bin','Pclass', 'Embarked']

train_dummies = pd.get_dummies(train[dummy_list], columns=dummy_list)

train.drop(labels = dummy_list, axis = 1, inplace = True)



# drop one column for each dummy variable to avoid the dummy trap!

dummy_drop = ['title_Miss', 'age_bin_(60.105, 90.0]', 'fare_bin_(26.0, 912.329]', 'Pclass_3', 'Embarked_Q']

train_dummies.drop(train_dummies[dummy_drop], axis =1, inplace=True)



train_dummies = train_dummies.rename(columns={'age_bin_(0.1, 20.315]':'age_bin_1', 'age_bin_(20.315, 40.21]': 'age_bin_2', 'age_bin_(40.21, 60.105]': 'age_bin_3', 'fare_bin_(-0.001, 8.662]':'fare_bin_1', 'fare_bin_(8.662, 26.0]':'fare_bin_2'})

train_final = pd.concat([train, train_dummies], axis=1)



train_final.head()

"""

This section of the code prepares the data from the test dataset for analysis

"""





# importing the .csv dataset as test

test = pd.read_csv('../input/titanic/test.csv')



# convert the Sex column to a binary format

test.loc[test['Sex'] == 'female', "Sex"] = 0

test.loc[test['Sex'] == 'male', "Sex"] = 1

test['Sex'] = test['Sex'].astype(int)



# figure out the median fare in 3rd class to assign to missign fare value

test.Fare.fillna(value=7.89, inplace=True)



# extract titles from the Name columns. Keep what's left of "." and right of ", "

test['title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



# replace special titles with Mr, Mrs and Miss

male_titles = ['Master','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']

test.title.replace(to_replace=male_titles, value='Mr', inplace = True)

test.title.replace(to_replace=['Mme', 'the Countess', 'Lady', 'Dona'], value='Mrs', inplace = True)

test.title.replace(to_replace=['Ms', 'Mlle'], value='Miss', inplace = True)



# Assign male/female doctors to Mr / Mrs

test.loc[(test.title == 'Dr') & (test.Sex == 1), 'title']='Mr'

test.loc[(test.title == 'Dr') & (test.Sex == 0), 'title']='Mrs'



# replace NaN values for the age column with the median age for the respective Pclass

test.loc[(test['Age'].isnull())&(test['SibSp'] >= 3), 'Age'] = 9  # Kids with siblings

test.loc[(test['Age'].isnull())&(test['Pclass']==1), 'Age'] = 37  # 1st class passengers

test.loc[(test['Age'].isnull())&(test['Pclass']==2), 'Age'] = 29  # 2nd class passengers

test.loc[(test['Age'].isnull())&(test['Pclass']==3), 'Age'] = 24  # 3rd class passengers



# cathegorize the Age column into PREDIFINED bins

age_scalars = [0.1, 20.315, 40.21,60.105, 90.0]

test['age_bin'] = pd.cut(test['Age'],bins=age_scalars)



# cathegorize the Fare column into PREDIFINED bins

fare_scalars = [-0.001, 8.662 , 26.0, 912.329]

test['fare_bin'] = pd.cut(test['Fare'],bins=fare_scalars)



# create new column family, which describes the family size

test['family'] = test.SibSp + test.Parch

# cathegorize the family column into binary format

test.loc[train['family']>0, 'family']=1



# drop columns not used for analysis

to_drop = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

test.drop(labels = to_drop, axis = 1, inplace= True)



# Generate dummy vectors for the  columns in dummy_list

dummy_list = ['title', 'age_bin', 'fare_bin','Pclass', 'Embarked']

test_dummies = pd.get_dummies(test[dummy_list], columns=dummy_list)

test.drop(labels = dummy_list, axis = 1, inplace = True)



# drop one column for each dummy variable to avoid the dummy trap!

dummy_drop = ['title_Miss', 'age_bin_(60.105, 90.0]', 'fare_bin_(26.0, 912.329]', 'Pclass_3', 'Embarked_Q']

test_dummies.drop(test_dummies[dummy_drop], axis =1, inplace=True)



test_dummies = test_dummies.rename(columns={'age_bin_(0.1, 20.315]':'age_bin_1', 'age_bin_(20.315, 40.21]': 'age_bin_2', 'age_bin_(40.21, 60.105]': 'age_bin_3', 'fare_bin_(-0.001, 8.662]':'fare_bin_1', 'fare_bin_(8.662, 26.0]':'fare_bin_2'})

test_final = pd.concat([test, test_dummies], axis=1)





test_final.head()

"""

This section of the code tests the performance of the listed ML alrogithms (default hyperparameters) on the data as it is.

"""







# Now, run several ML algorithms to see how they perform.

# The following list MLA of machine learning algorithms will be used:



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



# pre-define how to split the training data into a test/training set

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )



# create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



# create table to compare MLA predictions

Target = 'Survived'

data_columns = ['Sex','family','title_Mr','title_Mrs','age_bin_1','age_bin_2','age_bin_3','Pclass_1','Pclass_2','Embarked_C','Embarked_S', 'fare_bin_1','fare_bin_2'] # omit:

MLA_predict = train_final[Target]







# index through MLA and save performance to table

row_index = 0      # inititalizing the counting/indexing parameter

for alg in MLA:    # the MLAs are referred to as 'alg'



    # retrieve the name and parameters of the algorithm

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    # score each model with cross validation!

    cv_results = model_selection.cross_validate(alg, train_final[data_columns], train_final[Target], cv  = cv_split, return_train_score=True)

    

    # write the test outcome to the MLA_compare table

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(train_final[data_columns], train_final[Target])

    MLA_predict[MLA_name] = alg.predict(train_final[data_columns])

    

    row_index+=1



#print and sort the generated table

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
# plotting test results in seaborn

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')





plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
"""

Perform feature selection on the current dataset, using a DecisionTreeClassifier for model building.

In this case, Recursive Feature Elimination is performed (RFE) for feature selection.

This algorithm evaluates, which feature is the least important, 

drops it and reevaluates the model performance until the optimal number of features is reached.



It seems like this algorithm performs poorly in this situation.

In fact, model performance is only raised from 80.34% to 80.37% correct predictions.

"""



# Initialize and measure model performance

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, train_final[data_columns], train_final[Target], cv  = cv_split, return_train_score=True)

dtree.fit(train_final[data_columns], train_final[Target])





# Report the performance of the analysis before feature selection

print('BEFORE DT RFE Training Shape Old: ', train_final[data_columns].shape) 

print('BEFORE DT RFE Training Columns Old: ', train_final[data_columns].columns.values)



print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)



# Perform the actual feature selection

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(train_final[data_columns], train_final[Target])



# transform x&y to reduced features and fit new model

X_rfe = train_final[data_columns].columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, train_final[X_rfe], train_final[Target], cv  = cv_split, return_train_score=True)



# Report the performance of the analysis after feature selection

print('AFTER DT RFE Training Shape New: ', train_final[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))
"""

Use the updated column values 'X-rfe' to re-evaluate the performance of the ML algorithms listed above.



The performance was only marginally increased by the feature selection. 

On one hand, probably because the step was optimized for DecisionTreeClassifier. On the other,

Feature Selection did improve even the DecisionTreeClassifier that much.

"""







# index through MLA and save performance to table

row_index = 0      # inititalizing the counting/indexing parameter

for alg in MLA:    # the MLAs are referred to as 'alg'



    # retrieve the name and parameters of the algorithm

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    # score each model with cross validation!

    cv_results = model_selection.cross_validate(alg, train_final[X_rfe], train_final[Target], cv  = cv_split, return_train_score=True)

    

    # write the test outcome to the MLA_compare table

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(train_final[X_rfe], train_final[Target])

    MLA_predict[MLA_name] = alg.predict(train_final[X_rfe])

    

    row_index+=1



#print and sort the generated table

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
"""

This part of the code uses GridSearchCV to determine the most suited parameters for the DecisionTreeClassifier.

"""



# To tweak performance of the machine learning algorithms by varying the hyperparameters

# In this case, this is specifically done for the DecisionTreeClassifier

# This can also be done for multiple models that use similar hyperparameters, but it takes a lot of computing power.



# define which parameters should be applied and tested

param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best

              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator

             }



# measure performance before hyperparameter optimization

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, train_final[X_rfe], train_final[Target], cv  = cv_split, return_train_score=True)

dtree.fit(train_final[X_rfe], train_final[Target])



# report performance before optimization

print('BEFORE DT Parameters: ', dtree.get_params())

print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))

print('-'*10)





tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train_final[X_rfe], train_final[Target])



#print(tune_model.cv_results_.keys())

#print(tune_model.cv_results_['params'])

print('AFTER DT Parameters: ', tune_model.best_params_)

#print(tune_model.cv_results_['mean_train_score'])

print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(tune_model.cv_results_['mean_test_score'])

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

"""

Here, the DecisionTreeClassifier is run on the whole training dataset, 

using the hyperparameters and relevant columns determined above.

The data is exported and saved as a file named "submit.csv"

"""





# Run decision tree classifier on the train dataset using the determined parameters

dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=5, random_state = 0)

dtree.fit(train_final[X_rfe], train_final[Target])



# Re-import the original test data to retrieve ['PassengerId']

test_origin = pd.read_csv('../input/titanic/test.csv')



# Predict values for the test set, append the PassengerID column

submit=pd.Series(dtree.predict(test_final[X_rfe]), name='Survived')

submit=pd.concat([test_origin['PassengerId'], submit], axis = 1)



# Save the result for submission

submit.to_csv("../working/submit.csv", index=False)