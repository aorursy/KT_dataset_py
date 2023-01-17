import pandas as pd

from sklearn.model_selection import (GroupKFold, KFold, ShuffleSplit,

                                     StratifiedKFold,RepeatedKFold)
df = pd.DataFrame()

df['y'] = [

    0, 0, 0,

    1, 1, 1,

]



df['sex'] = [

    'male', 'female',

    'male', 'female',

    'male', 'female',

]



df['rank'] = [

    1,1,

    2,2,

    3,3,

]



df
X = df[['sex', 'rank']]

y = df[['y']]
# check data order and class blance

# Provides train/test indices to split data in train/test sets.

# Split dataset into k consecutive folds (without shuffling by default).

kf = KFold(n_splits=3)



for train_index, valid_index in kf.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
# check data order

# Provides train/test indices to split data in train/test sets.

# Split dataset into k consecutive folds (without shuffling by default).

kf = KFold(n_splits=3,

           random_state=0)



for train_index, valid_index in kf.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
# with shuffling

kf = KFold(n_splits=3,

           random_state=0,

           shuffle=True)



for train_index, valid_index in kf.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
skf = StratifiedKFold(n_splits=3,

                      random_state=0,

                      shuffle=True)



for train_index, valid_index in skf.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
gkf = GroupKFold(n_splits=2)



groups = [0, 1,

          0, 1,

          0, 1]



for train_index, valid_index in gkf.split(X, y, groups):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
# grouping male/female 

gkf = GroupKFold(n_splits=3)



groups = [0, 0,

          1, 1,

          2, 2]



for train_index, valid_index in gkf.split(X, y, groups):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
X
gkf = GroupKFold(n_splits=2)



rank_groups = [

    0,1,2,

    0,1,2,

]



for train_index, valid_index in gkf.split(X, y, groups=rank_groups):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
ss = ShuffleSplit(n_splits=2, 

                  train_size=0.50,

                  random_state=0)



for train_index, valid_index in ss.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
ss = ShuffleSplit(n_splits=3, 

                  train_size=0.50,

                  random_state=0)



for train_index, valid_index in ss.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
ss = ShuffleSplit(n_splits=2,

                  train_size=0.67,

                  random_state=0)



for train_index, valid_index in ss.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
rkf = RepeatedKFold(n_splits=3,

                    n_repeats=1,

                    random_state=0)



for train_index, valid_index in rkf.split(X, y):

    print(train_index, valid_index)

    print(df.iloc[train_index])

    print('------------------')
rkf = RepeatedKFold(n_splits=3,

                    n_repeats=2,

                    random_state=0)



for train_index, valid_index in rkf.split(X, y):

    print(train_index, valid_index)

    print('------------------')