# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')

def append_structures(df):

    # JOIN on atom_index_0

    df = pd.merge(df, structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])

    df = df.rename({'x': 'x_0', 'y': 'y_0', 'z': 'z_0', 'atom': 'atom_0'}, axis=1)

    df = df.drop(['atom_index', 'atom_index_0'], axis=1)



    # JOIN on atom_index_1

    df = pd.merge(df, structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

    df = df.rename({'x': 'x_1', 'y': 'y_1', 'z': 'z_1', 'atom': 'atom_1'}, axis=1)

    df = df.drop(['atom_index', 'atom_index_1'], axis=1)

    

    return df
def add_molecule_features(df):

    foo = pd.DataFrame({'1JHC_total': X.groupby('molecule_name')['1JHC'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'1JHN_total': X.groupby('molecule_name')['1JHN'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'2JHC_total': X.groupby('molecule_name')['2JHC'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'2JHN_total': X.groupby('molecule_name')['2JHN'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'2JHH_total': X.groupby('molecule_name')['2JHH'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'3JHC_total': X.groupby('molecule_name')['3JHC'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    foo = pd.DataFrame({'3JHH_total': X.groupby('molecule_name')['3JHH'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])

    

    foo = pd.DataFrame({'3JHN_total': X.groupby('molecule_name')['3JHN'].sum()})

    foo['molecule_name'] = foo.index

    df = pd.merge(df, foo, on = ['molecule_name'])



    return df

    

def set_features(df):

    df = append_structures(df)

    

    ### Feature engineering goes here

    

    # For now drop non-numeric

    df = df.drop(['atom_0'], axis=1)

    df = df.drop(['atom_1'], axis=1)

    

    df = df.drop(['molecule_name'], axis=1)

    dummies = pd.get_dummies(df['type'])

    df = pd.concat([df.drop(['type'], axis=1), dummies], axis=1)

    return df

    
from sklearn.ensemble import RandomForestRegressor



regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)



Y = train['scalar_coupling_constant']

X = train.drop(['scalar_coupling_constant'], axis=1)



X = set_features(X)



regr.fit(X.head(10000),Y.head(10000))

Y_hat = regr.predict(X)



test_X = test

test_X = set_features(test_X)

test_Y = regr.predict(test_X)
submission_df = pd.DataFrame({'id': test_X['id'], 'scalar_coupling_constant': test_Y})



submission_df.to_csv('submission.csv', index=False)
!wc -l submission.csv
