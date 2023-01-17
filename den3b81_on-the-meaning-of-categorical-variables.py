### load modules

import pandas as pd

import numpy as np
# load training data (we could also include the test data in our analysis, it won't change much)

train_df  = pd.read_csv('../input/train.csv')



# remove ID, y and constant columns 

df = train_df.drop(['ID','y'], axis = 1)

df = df.loc[:, (df != df.ix[0]).any()] 
# now let's loop across the categorical variables

categorical = ['X0','X1','X2','X3','X4','X5','X6','X8']

for cat in categorical:   

    # this groupby finds the columns which are constant within classes in the categorical feature

    temp = (df.groupby(cat).std().mean()==0)    

    constant_cols = temp[temp==True].index.tolist()

    print('{1} constant columns across {0}\n'.format(cat,len(constant_cols)))

    print(constant_cols)

    print('********************************')
# let's see for instance the columns which are constant across X0 (taken from above)

const_cols_across_X0 = ['X29', 'X54', 'X76', 'X118', 'X119', 'X136', 'X186', 

                         'X187', 'X194', 'X231', 'X232', 'X236', 'X263', 'X277', 

                         'X279', 'X313', 'X314', 'X315', 'X316']

df.groupby('X0').mean()[const_cols_across_X0]
# load test data

test_df  = pd.read_csv('../input/test.csv')



# remove ID, y, combine datasets and remove constant columns 

df = pd.concat([train_df.drop(['ID','y'], axis = 1),test_df.drop(['ID'], axis = 1)]).reset_index(drop = True)

df = df.loc[:, (df != df.ix[0]).any()] 
# now let's loop across the categorical variables

categorical = ['X0','X1','X2','X3','X4','X5','X6','X8']

for cat in categorical:   

    # this groupby finds the columns which are constant within classes in the categorical feature

    temp = (df.groupby(cat).std().mean()==0)    

    constant_cols = temp[temp==True].index.tolist()

    print('{1} constant columns across {0}\n'.format(cat,len(constant_cols)))

    print(constant_cols)

    print('********************************')