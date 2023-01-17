import numpy as np

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
def test_mean_target_encoding(train, test, target, categorical, alpha=5):

    # Calculate global mean on the train data

    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties

    train_groups = train.groupby(categorical)

    category_sum = train_groups[target].sum()

    category_size = train_groups.size()



    # Calculate smoothed mean target statistics

    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

    

    # Apply statistics to the test data and fill new categories

    test_feature = test[categorical].map(train_statistics).fillna(global_mean)

    return test_feature.values
def train_mean_target_encoding(train, target, categorical, alpha=5):

    # Create 5-fold cross-validation

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, random_state=123, shuffle=True)

    train_feature = pd.Series(index=train.index)

    

    # For each folds split

    for train_index, test_index in kf.split(train):

        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

      

        # Calculate out-of-fold statistics and apply to cv_test

        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)

        

        # Save new feature for this particular fold

        train_feature.iloc[test_index] = cv_test_feature       

    return train_feature.values
def mean_target_encoding(train, test, target, categorical, alpha=5):

  

    # Get the train feature

    train_feature = train_mean_target_encoding(train, target, categorical, alpha)

  

    # Get the test feature

    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)

    

    # Return new features to add to the model

    return train_feature, test_feature
train_pclass_enc, test_pclass_enc = mean_target_encoding(train,test,'Survived','Pclass')
train['Pclass_enc'] = train_pclass_enc

test['Pclass_enc'] = test_pclass_enc
train.drop('Pclass', axis=1,inplace=True)

test.drop('Pclass', axis=1,inplace=True)
display(train.head(3))

display(test.head(3))