import numpy as np

import pandas as pd

import warnings



warnings.simplefilter('ignore')



train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
card_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
train[card_features].head()
pd.concat([train[card_features].isna().sum(), test[card_features].isna().sum()], axis=1).rename(columns={0: 'train_NaNs', 1: 'test_NaNs'})
pd.concat([train[card_features].isna().sum() / train.shape[0], test[card_features].isna().sum() / test.shape[0]], axis=1).rename(columns={0: 'train_NaNs_%', 1: 'test_NaNs_%'})
#Some usefull functions



def count_uniques(train, test, pair):

    unique_train = []

    unique_test = []



    for value in train[pair[0]].unique():

        unique_train.append(train[pair[1]][train[pair[0]] == value].value_counts().shape[0])



    for value in test[pair[0]].unique():

        unique_test.append(test[pair[1]][test[pair[0]] == value].value_counts().shape[0])



    pair_values_train = pd.Series(data=unique_train, index=train[pair[0]].unique())

    pair_values_test = pd.Series(data=unique_test, index=test[pair[0]].unique())

    

    return pair_values_train, pair_values_test



def fill_card_nans(train, test, pair_values_train, pair_values_test, pair):

    print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )

    print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )



    print('Filling train...')

    

    for value in pair_values_train[pair_values_train == 1].index:

        train[pair[1]][train[pair[0]] == value] = train[pair[1]][train[pair[0]] == value].value_counts().index[0]

        

    print('Filling test...')



    for value in pair_values_test[pair_values_test == 1].index:

        test[pair[1]][test[pair[0]] == value] = test[pair[1]][test[pair[0]] == value].value_counts().index[0]

        

    print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )

    print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )

    

    return train, test



def nans_distribution(train, test, unique_train, unique_test, pair):

    train_nans_per_category = []

    test_nans_per_category = []



    for value in unique_train.unique():

        train_nans_per_category.append(train[train[pair[0]].isin(list(unique_train[unique_train == value].index))][pair[1]].isna().sum())



    for value in unique_test.unique():

        test_nans_per_category.append(test[test[pair[0]].isin(list(unique_test[unique_test == value].index))][pair[1]].isna().sum())



    pair_values_train = pd.Series(data=train_nans_per_category, index=unique_train.unique())

    pair_values_test = pd.Series(data=test_nans_per_category, index=unique_test.unique())

    

    return pair_values_train, pair_values_test
train[train['card1'] == 13926][['card1', 'card2']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'card2'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'card2'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'card2'))
train[train['card1'] == 13926][['card1', 'card3']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'card3'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'card3'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'card3'))
train[train['card1'] == 13926][['card1', 'card4']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'card4'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'card4'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'card4'))
train[train['card1'] == 13926][['card1', 'card5']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'card5'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'card5'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'card5'))
train[train['card1'] == 13926][['card1', 'card6']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'card6'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'card6'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'card6'))
pd.concat([train[card_features].isna().sum(), test[card_features].isna().sum()], axis=1).rename(columns={0: 'train_NaNs', 1: 'test_NaNs'})
train[card_features].head()
print('Card3 == 150: ', train[train['card3'] == 150]['card2'].nunique())

print('Card4 == mastercard: ', train[train['card4'] == 'mastercard']['card2'].nunique())

print('Card5 == 102: ', train[train['card5'] == 102]['card2'].nunique())

print('Card6 == credit: ', train[train['card6'] == 'credit']['card2'].nunique())
print('Card2 == 327: ', train[train['card2'] == 327]['card5'].nunique())

print('Card3 == 150: ', train[train['card3'] == 150]['card5'].nunique())

print('Card4 == mastercard: ', train[train['card4'] == 'mastercard']['card5'].nunique())

print('Card6 == credit: ', train[train['card6'] == 'credit']['card5'].nunique())
train[train['card1'] == 13926][['card1', 'addr2']]
unique_values_train, unique_values_test = count_uniques(train, test, ('card1', 'addr2'))

pd.concat([unique_values_train.value_counts(), unique_values_test.value_counts()], axis=1).rename(columns={0: 'train', 1: 'test'})
train_nan_dist, test_nan_dist = nans_distribution(train, test, unique_values_train, unique_values_test, ('card1', 'addr2'))

pd.concat([train_nan_dist, test_nan_dist], axis=1).rename(columns={0: 'train', 1: 'test'})
train, test = fill_card_nans(train, test, unique_values_train, unique_values_test, ('card1', 'addr2'))
train[train['card1'] == 13926]['addr2'].value_counts().shape[0] == 1
depend_features = []



for col in train.columns:

    if train[train['card1'] == 13926][col].value_counts().shape[0] == 1:

        depend_features.append(col)



print(depend_features)
def fill_pairs(train, test, pairs):

    for pair in pairs:



        unique_train = []

        unique_test = []



        print(f'Pair: {pair}')

        print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )

        print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )



        for value in train[pair[0]].unique():

            unique_train.append(train[pair[1]][train[pair[0]] == value].value_counts().shape[0])



        for value in test[pair[0]].unique():

            unique_test.append(test[pair[1]][test[pair[0]] == value].value_counts().shape[0])



        pair_values_train = pd.Series(data=unique_train, index=train[pair[0]].unique())

        pair_values_test = pd.Series(data=unique_test, index=test[pair[0]].unique())

        

        print('Filling train...')



        for value in pair_values_train[pair_values_train == 1].index:

            train.loc[train[pair[0]] == value, pair[1]] = train.loc[train[pair[0]] == value, pair[1]].value_counts().index[0]



        print('Filling test...')



        for value in pair_values_test[pair_values_test == 1].index:

            test.loc[test[pair[0]] == value, pair[1]] = test.loc[test[pair[0]] == value, pair[1]].value_counts().index[0]



        print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )

        print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )

        

    return train, test
pairs = [('card1', 'card2'), ('card1', 'card3')]



train, test = fill_pairs(train, test, pairs)