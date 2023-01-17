# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_X = pd.read_csv("../input/act_train.csv", sep=",")

people = pd.read_csv('../input/people.csv', sep=',')
people.columns = ['people_id']+[f+'_ppl' for f in people.columns if f not in ['people_id']]

fold_X = pd.merge(train_X, people, on='people_id', how='left', suffixes=['_act', ''])[['activity_id','group_1_ppl']]

del people

del train_X
n_folds = 5

fold_X.sort_values(by='group_1_ppl', inplace=True)

# Check group_1 values occurences to drop groups that have less than n_folds occurences

group_counts = fold_X.group_1_ppl.value_counts()

group_counts_merge = group_counts.reset_index()

group_counts_merge.columns = ['group_1_ppl','counts']

# Merge occurences with data

fold_X = pd.merge(fold_X, group_counts_merge, on='group_1_ppl', how='left')

fold_X.sort_values(by='counts', inplace=True)

# Make sure index is 0,1,2...n

fold_X.reset_index(inplace=True, drop=True)
fold_X['fold'] = fold_X.index % 5

fold_X['new_index'] = np.floor(fold_X.index/5).astype(int)

test_folds = fold_X.pivot(index = 'new_index', columns='fold', values='activity_id')

test_folds.drop(439458,axis=0,inplace=True) # The last row contains NaN
nogroup_len = 65000

f_idx = fold_X.index

for i in range(n_folds):

    test_folds.ix[:(nogroup_len-1),i] =  fold_X.ix[f_idx[nogroup_len*i:nogroup_len*(i+1)], "activity_id"].values
print(test_folds.tail())

test_folds.to_csv('test_folds.csv', index=False, sep=',')
nb_samples = len(test_folds)

train_folds = pd.DataFrame(np.zeros((nb_samples * 4, 5)))

for i in range(5):

    g = 0

    for n in range(5):

        if n != i:

            train_folds.ix[g * nb_samples:(g + 1) * nb_samples - 1, i] = test_folds.ix[:, n].values

            g += 1

train_folds.to_csv('train_folds.csv')
train_folds.head()