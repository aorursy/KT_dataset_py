# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading data. Takes time to load due to data size.



train = pd.read_csv('../input/train.csv/train.csv')

test = pd.read_csv('../input/test.csv/test.csv')

print (train.shape, test.shape)
# Merging datasets to apply common transformations



train['train_identifier'] = 1

test['train_identifier'] = 0

combined = pd.concat([train, test], ignore_index=True)

print (combined.shape)



# Deleting redundant dataframes

del train, test

gc.collect()
# Identify and remove columns from dataset which are categorical in nature and have more than 100 categories --> that is

# probably useless for our predictions



combined_cols = combined.columns

redundant_cols = []



for col in combined.columns:

    unique_values = len(set(combined[col]))

    if isinstance(combined[col][0], str) and unique_values >= 100:

#         print (col, combined[col][0], unique_values)

        redundant_cols.append(col)



print (redundant_cols)



combined.drop(redundant_cols, axis=1, inplace=True)

print (combined.shape)
# Imputation of missing values



# (a) Filling columns with single dominant category (>= 85%) with "MODE"



cols_with_monopoly = ['x1', 'x2', 'x11', 'x13', 'x14', 'x25', 'x32', 'x33', 'x42', 'x44', 'x45', 'x56', 'x62', 'x63', 'x72',

                      'x74', 'x75', 'x86', 'x92', 'x93', 'x102', 'x104', 'x105', 'x116', 'x127', 'x129', 'x141']

for col in cols_with_monopoly:

    combined[col].fillna(value='NO', inplace=True)
# (b) Imputing missing values of categorical features using continuous features



# Creating a list of predictors



cat_features = []

for col in combined.columns:

    if isinstance(combined[col][0], str):

        cat_features.append(col)

print ('# of cat features: ', len(cat_features))



cols_with_missing_values = ['x10', 'x12', 'x24', 'x26', 'x41', 'x43', 'x55', 'x57', 'x71', 'x73', 'x85', 'x87', 'x101', 'x103',

                            'x115', 'x117', 'x126', 'x128', 'x130', 'x140', 'x142']



reduced_cat_features = [col for col in cat_features if col not in cols_with_missing_values]

print ('# of reduced cat features: ', len(reduced_cat_features))



combined = pd.get_dummies(combined, prefix=reduced_cat_features, columns=reduced_cat_features)

print (len(combined.columns))

predictors = [col for col in combined.columns if col not in cols_with_missing_values]

predictors.remove('id')

predictors.remove('train_identifier')

print ('Predictors: ', len(predictors), predictors)
# Imputing missing values using classifier (hyperparameters not tuned)

# Downloading resultant dataframe for further use



from sklearn.linear_model import SGDClassifier



classifier = SGDClassifier(verbose=0, n_jobs=8, random_state=9)



df = combined.copy(deep=True)



for col in cols_with_missing_values:

    print ('Imputation starts for: ', col, df.shape)

    df_train = df[df[col].notnull()]

    df_test = df[~ df[col].notnull()]

    print (df_train.shape, df_test.shape)

    model = classifier.fit(df_train[predictors], df_train[col])

    df_test[col] = model.predict(df_test[predictors])

    df = pd.concat([df_train, df_test])



print ('Imputation completed')



# df.to_pickle('data_frame_with_imputed_values.pickle')

# print ('Download completed')