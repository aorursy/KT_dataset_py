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
## import data 

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()

train.describe()

test.info()

#Create family_size column

train['Family_Size']=train['SibSp']+train['Parch']

test['Family_Size']=test['SibSp']+test['Parch']
#Define variables

y = train.Survived

X = train.drop(['PassengerId','Survived'], axis=1)



test_for_submission = test.drop(['PassengerId'], axis=1) #This is the df that will be using for the predictions at the end



#split data in train and validation

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Work out categorical columns 

categorical_cols = [col for col in X if X[col].dtype =='object']



object_nunique = list(map(lambda col: X_train[col].nunique(), categorical_cols))

d = dict(zip(categorical_cols, object_nunique))



object_nunique = list(map(lambda col: test_for_submission[col].nunique(), categorical_cols))

d1 = dict(zip(categorical_cols, object_nunique))



# Print number of unique entries by column, in ascending order

print(sorted(d.items(), key=lambda x: x[1]))

print(sorted(d1.items(), key=lambda x: x[1]))



#Check which columns have low cardinality

low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() < 10]

high_cardinality_cols =list(set(categorical_cols)-set(low_cardinality_cols))

print(low_cardinality_cols)
##Columns to be encoded wuth dummies, I keep only the ones with low cardinality.

drop_X_train = X_train.drop(high_cardinality_cols, axis=1)

drop_X_valid = X_valid.drop(high_cardinality_cols, axis=1)



drop_train_for_submission = X.drop(high_cardinality_cols, axis=1) #This is the df that will be using for the training at the end

drop_test_for_submission = test_for_submission.drop(high_cardinality_cols, axis=1) #This is the df that will be using for the predictions at the end



##Dummied data sets

dummied_drop_X_train = pd.get_dummies(drop_X_train)

dummied_drop_X_valid = pd.get_dummies(drop_X_valid)



dummied_drop_train_for_submission = pd.get_dummies(drop_train_for_submission) #This is the df that will be using for the training at the end

dummied_drop_test_for_submission = pd.get_dummies(drop_test_for_submission) #This is the df that will be using for the predictions at the end
## Deal with missing values imputation

X_train_plus = dummied_drop_X_train.copy()

X_valid_plus = dummied_drop_X_valid.copy()



train_for_submission_plus = dummied_drop_train_for_submission.copy() #This is the df that will be using for the training at the end

test_for_submission_plus = dummied_drop_test_for_submission.copy() #This is the df that will be using for the predictions at the end



#Because I am going to add a column to indicate which values where imputed, I have to add the same amount of cols to all df

cols_with_missing = []

for ds in [dummied_drop_X_train, dummied_drop_X_valid, dummied_drop_test_for_submission]:

    cols_with_missing_ds = [col for col in ds.columns 

                          if ds[col].isnull().any()]

    if len(cols_with_missing)<len(cols_with_missing_ds):

        cols_with_missing = cols_with_missing_ds

print(cols_with_missing)



# Make new columns indicating what values will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    

    train_for_submission_plus[col + '_was_missing'] = train_for_submission_plus[col].isnull() #This is the df that will be using fo

    test_for_submission_plus[col + '_was_missing'] = test_for_submission_plus[col].isnull() #This is the df that will be using for the predictions at the end



# Impute missing values 

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



imputed_train_for_submission_plus = pd.DataFrame(my_imputer.transform(train_for_submission_plus)) #This is the df that will be using for the training at the end

imputed_test_for_submission_plus = pd.DataFrame(my_imputer.transform(test_for_submission_plus))#This is the df that will be using for the predictions at the end



#Label columns back, after imputation

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns





imputed_train_for_submission_plus.columns = train_for_submission_plus.columns #This is the df that will be using for the training at the end

imputed_test_for_submission_plus.columns = test_for_submission_plus.columns #This is the df that will be using for the predictions at the end
#Create the data sets after the whole process to be used in the models

X_train_clean = imputed_X_train_plus

X_valid_clean = imputed_X_valid_plus



#Random forest

from sklearn.ensemble import RandomForestClassifier



#define a function to test different values of the same parameter

def get_score(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestClassifier(n_estimators = 100,max_depth=50, max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    #preds_val = model.predict(val_X)

    train_acc = model.score(train_X, train_y)

    test_acc = model.score(val_X, val_y)

    print("Max leaf nodes: ", max_leaf_nodes, "score_test: ", test_acc, '/ test_train: ', train_acc)

    return(test_acc)

#Try diffetent values for max_leaf_nodes and pick the best one.

candidate_max_leaf_nodes = [5,10,20,50,100,120,140,150,200,500,1000]    

scores = {leaf_size: get_score(leaf_size, X_train_clean, X_valid_clean, y_train, y_valid) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = max(scores, key=scores.get)

print("best: ", best_tree_size)


model = RandomForestClassifier(n_estimators=100, max_depth=50, max_leaf_nodes=best_tree_size, random_state=1)

#model.fit(imputed_train_for_submission_plus, y)

model.fit(X_train_clean, y_train)

predictions = model.predict(imputed_test_for_submission_plus)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")