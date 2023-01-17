import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
training_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# Understanding data structure

training_data.head(3)

print('training_data.info():::')

training_data.info()
# Quick overview missed and object data type

# Check to see if there is any missing data columns

# In training data

cols_in_training_data_with_nan = [col for col in training_data.columns

                        if training_data[col].isnull().any()] 

# print('cols_in_training_data_with_nan :', cols_in_training_data_with_nan)



# In Test data

cols_in_testing_data_with_nan = [col for col in test_data.columns

                        if test_data[col].isnull().any()] 



# print('cols_in_testing_data_with_nan: ', cols_in_testing_data_with_nan)
# To find the difference between two list values

# set(cols_in_training_data_with_nan)-set(cols_in_testing_data_with_nan)
all_missing_cols = set(cols_in_training_data_with_nan + cols_in_testing_data_with_nan)

#finding the Nan count in test set

print('Printing training and test data nan count...')

for idx, col in enumerate(all_missing_cols):

#     ToDo: populate the values in dataframe

    print(idx, ': ', col , ' : ',training_data[col].isnull().sum(),'   ;', test_data[col].isnull().sum())
# mylist = set(training_data.dtypes)  # This will all the unique dtypes in the dataframe

exclude_types ={np.int64, np.float64}

# categories give object type columns

categories = list(training_data.select_dtypes(exclude=exclude_types).columns)

print('categorical columns : ', categories)
# From above analysis ['MiscFeature','Fence','Alley','PoolQC'] has loads of NAN..so removing them

cols_tobe_removed = ['MiscFeature','Fence','Alley','PoolQC']

training_data = training_data.drop(cols_tobe_removed, axis=1)

test_data = test_data.drop(cols_tobe_removed, axis=1)
def fill_nan_in_numerical(col):

    total_nans = training_data[col].isnull().sum()

    

    training_data[col].fillna(training_data[col].median() , inplace = True)

    # Cause test_data dont have target columns

    if col !='SalePrice':

        test_data[col].fillna(test_data[col].median() , inplace = True)

    if total_nans > 0:

        print(col , 'nan count in train data before func call: ' , total_nans)

        print(col , 'nan count in train data after func call: ' , training_data[col].isnull().sum())

exclude_type ={np.object}

# categories give object type columns

numerical_cols = list(training_data.select_dtypes(exclude=exclude_type).columns)

print('numerical_cols : ', numerical_cols)
for col in numerical_cols:

    fill_nan_in_numerical(col)
training_data.shape

test_data.shape
# function to draw freq distribution of categorical column

def freq_dist_for(category):

    plt.style.use('ggplot')

    fig, (axis1, axis2) = plt.subplots(1, 2, sharex = True, figsize=(20,5))

    g1 = sns.countplot(x=training_data[category],ax=axis1)

    g2 = sns.countplot(x=test_data[category],ax=axis2)

    axis1.set_xticklabels(axis1.get_xticklabels(), rotation = 45)

    axis2.set_xticklabels(axis2.get_xticklabels(), rotation = 45)

    plt.show()
freq_dist_for('FireplaceQu')

# Tip: you can draw freq distribution for all cateogries by following command:

# Uncomment following code to see the plots...

# for category in categories:

#     freq_dist_for(category)
# function to fill the nan's with high frequency values:

def fill_nans_in_cat_col(category, value):

    """function to fill the nan's with high frequency value"""

    total_nans = training_data[category].isnull().sum()

    training_data[category].fillna(value, inplace = True)

    test_data[category].fillna(value, inplace = True)

    if total_nans > 0:

        print(col , 'nan count in train data before func call: ' , total_nans)

        print(col , 'nan count in train data after func call: ' , training_data[category].isnull().sum())

fill_nans_in_cat_col('FireplaceQu', 'Gd')
def create_dummies(category):

    """Does following Things: 

        1. Creates dummy variables on both training and test

        2. Remove one feature column from dummy variables to remove the redundancy

        3. Add dummies variable to the data sets

        4. Remove original column from the data sets"""

    global training_data

    global test_data

    if category in training_data and category in test_data:

        high_freq_value_in_train_category = training_data[category].value_counts(normalize=True)

        #     train_idx = high_freq_value_in_train_category[high_freq_value_in_train_category <  high_freq_value_in_train_category.mean()].index

        train_idx = high_freq_value_in_train_category.head(2).index

        

        training_dummies =  pd.get_dummies(training_data[category])

        test_dummies = pd.get_dummies(test_data[category])

        for idx in train_idx:

            if idx in training_dummies and idx in test_dummies:

                training_data = pd.concat([training_data, training_dummies[idx]], axis=1)

                test_data = pd.concat([test_data, test_dummies[idx]], axis=1)



        training_data.drop(category, axis=1, inplace=True)

        test_data.drop(category, axis=1, inplace=True)

    else:

        print(category , ' Doesn\'t exist in both train and test data')

        if category in training_data:

            training_data.drop(category, axis=1, inplace=True)

        if category in test_data:

            test_data.drop(category, axis=1, inplace=True)

    
remaining_categories = set(categories) - set(cols_tobe_removed)

print('remaining_categories :', remaining_categories)

for category in remaining_categories:

    freq_dist_for(category)
# One shot encoding of the categorical values:

for category in remaining_categories:

#     print('processing :', category)

    create_dummies(category)

print('Done converting categorical data into dummies ...')    
training_data.shape

test_data.shape
# Separate target from training set

X_train = training_data.drop('SalePrice', axis=1)

y_train = training_data['SalePrice']



X_test = test_data
len(list(X_train.columns)), len(list(X_test.columns))

len(set(X_train.columns)), len(set(X_test.columns))

set(X_train.columns)-set(X_test.columns)

X_train.shape, X_test.shape

y_train.shape
# Correlation Plot

# ToDo: Commented cause it is taking long time to run 

# colormap = plt.cm.viridis

# plt.figure(figsize=(14,12))

# plt.title('Pearson Correlation of Features', y=1.05, size =15)

# sns.heatmap(training_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

#             square=True, cmap=colormap, linecolor='white', annot=True)

# plt.show()
logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

logistic_regression.score(X_train, y_train)
# Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_train, y_train)
# Random Forests

random_forest = RandomForestClassifier()

random_forest.fit(X_train,y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
sub = pd.DataFrame()

sub['Id'] = test_data['Id']

sub['SalePrice'] = y_pred

sub.to_csv('submission.csv',index=False)
# GaussianNB

gaussian_nb = GaussianNB()

gaussian_nb.fit(X_train,y_train)

y_pred = gaussian_nb.predict(X_test)

gaussian_nb.score(X_train, y_train)