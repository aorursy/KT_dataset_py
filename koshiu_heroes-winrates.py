import numpy as np 

import pandas as pd 

import os

import warnings 

warnings.filterwarnings('ignore')       

from sklearn.feature_extraction.text import CountVectorizer

from itertools import combinations

from scipy.sparse import hstack, csr_matrix



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'

SEED = 17
# Train dataset

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 

                                   index_col='match_id_hash')



y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})

y_train.reset_index(drop=True,inplace=True)

# Test dataset

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')

# from https://www.kaggle.com/kuzand kernel

def bag_of_heroes(df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr'):

    '''

    Bag of Heroes. Returns a csr matrix (+ list of feature names) or dataframe where each column represents

    a hero (ID) and each row represents a match.

    

    The value of a cell (i, j) in the returned matrix is:

        cell[i, j] = 0, if the hero or combination of heroes of the j-th column is not present in the i-th match

        cell[i, j] = r_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Radiant team,

        cell[i, j] = d_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Dire team,

        cell[i, j] = r_d_val, if the combination of heroes of the j-th column is between the Radiant and Dire teams (N>1, anti-synergy).

    

    Parameters:

    -----------

        df: dataframe with hero IDs, with columns ['r1_hero_id', ..., 'r5_hero_id', 'd1_hero_id', ..., 'd5_hero_id']

        N: integer 1 <= N <= 10, for N heroes combinations

        return_as: 'csr' for scipy csr sparse matrix, 'df' for pandas dataframe

    '''

    if N < 1 or N > df.shape[1]:

        raise Exception(f'The number N of hero-combinations should be 1 <= N <= {df.shape[1]}')

        

    # Convert the integer IDs to strings of the form id{x}{x}{x}

    df = df.astype(str).applymap(lambda x: 'id' + '0'*(3 - len(x)) + x)

    

    # Create a list of all hero IDs present in df

    hero_ids = np.unique(df).tolist()



    # Break df into teams Radiant (r) and Dire (d)

    df_r = df[[col for col in df.columns if col[0] == 'r']]

    df_d = df[[col for col in df.columns if col[0] == 'd']]

    

    # Create a list of all the hero IDs in df, df_r and df_d respectively

    f = lambda x: ' '.join(['_'.join(c) for c in combinations(sorted(x), N)])

    

    df_list = df.apply(f, axis=1).tolist()

    df_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))



    df_r_list = df_r.apply(f, axis=1).tolist()

    df_r_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    

    df_d_list = df_d.apply(f, axis=1).tolist()

    df_d_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    

    # Create countvectorizers

    vectorizer = CountVectorizer()

    vectorizer_r = CountVectorizer()

    vectorizer_d = CountVectorizer()

    

    X = vectorizer.fit_transform(df_list)[:-1]

    X_r = vectorizer_r.fit_transform(df_r_list)[:-1]

    X_d = vectorizer_d.fit_transform(df_d_list)[:-1]

    X_r_d = (X - (X_r + X_d))  

    X = (r_val * X_r + d_val * X_d + r_d_val * X_r_d)

    

    feature_names = vectorizer.get_feature_names()

    

    if return_as == 'csr':

        return X, feature_names

    elif return_as == 'df':

        return pd.DataFrame(X.toarray(), columns=feature_names, index=df.index).to_sparse(0)
train_boh = df_train_features.copy()

test_boh = df_test_features.copy()



df_full_features = pd.concat([train_boh, test_boh])



# Index to split the training and test data sets

idx_split = train_boh.shape[0]



heroes_df = df_full_features[[f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]]
%%time

boh = bag_of_heroes(heroes_df, N=1, r_val=1, d_val=-1, return_as='csr')[0]



X_heroes_train = boh[:idx_split]

X_heroes_test  = boh[idx_split:]
def win_rates(X_heroes_train, X_heroes_test,target):

    # Creating 115 columns of heroes, there '1' - Radiant hero,'-1' - Dire hero 

    X_heroes_train = pd.DataFrame(X_heroes_train.toarray(),

            columns=['f'+str(i) for i in range(X_heroes_train.shape[1])])



    X_heroes_test = pd.DataFrame(X_heroes_test.toarray(),

            columns=['f'+str(i) for i in range(X_heroes_test.shape[1])])



    heroes_target = target.reset_index(drop=True)





    # count win_prob for every hero ((win as d + win as r)/ all played games by this hero )

    X_heroes_train['radiant_win'] = heroes_target['radiant_win']



    

    hero_win_dict = dict()

    for i in range(0,115): # 115 Heroes

        hero_name = 'f'+str(i)



        wins_by_dire = X_heroes_train.radiant_win[(X_heroes_train['f'+str(i)]==-1) 

                                        & (X_heroes_train['radiant_win']==0) ].value_counts().get_values()

        wins_by_radiant = X_heroes_train.radiant_win[(X_heroes_train['f'+str(i)]==1) 

                                        & (X_heroes_train['radiant_win']==1) ].value_counts().get_values()

        total_games_by_hero = X_heroes_train.radiant_win[(X_heroes_train['f'+str(i)]==1) 

                                        | (X_heroes_train['f'+str(i)]==-1) ].value_counts().get_values().sum()

        hero_win_prob = (wins_by_dire+wins_by_radiant)/total_games_by_hero

        hero_win_dict[hero_name] = hero_win_prob # {hero: hero_winrate}



    # drop radinat_win

    X_heroes_train.drop(columns=['radiant_win'],inplace=True)

    

    # Now let's count winrate for each team for every game.

    

    # train

    r_win_prob = list()

    d_win_prob = list()

    radiant_match_winrate = 0

    dire_match_winrate = 0

    for x in range(0,X_heroes_train.shape[0]):

        radiant_match_winrate=0

        dire_match_winrate=0

        r_5_winrates = np.argwhere(X_heroes_train.loc[x].to_numpy()>0).flatten() 

        d_5_winrates = np.argwhere(X_heroes_train.loc[x].to_numpy()<0).flatten()  

        for x in r_5_winrates:

            radiant_match_winrate+=float(hero_win_dict['f'+str(x)])

        for y in d_5_winrates:

            dire_match_winrate+=float(hero_win_dict['f'+str(y)])



        r_win_prob.append(radiant_match_winrate/5)

        d_win_prob.append(dire_match_winrate/5)



    # test

    r_win_prob_test = list()

    d_win_prob_test = list()

    radiant_match_winrate = 0

    dire_match_winrate = 0

    for x in range(0,X_heroes_test.shape[0]):

        radiant_match_winrate = 0

        dire_match_winrate = 0

        r_5_winrates = np.argwhere(X_heroes_test.loc[x].to_numpy()>0).flatten()  

        d_5_winrates = np.argwhere(X_heroes_test.loc[x].to_numpy()<0).flatten()  

        for x in r_5_winrates:

            radiant_match_winrate += float(hero_win_dict['f'+str(x)])

        for y in d_5_winrates:

            dire_match_winrate += float(hero_win_dict['f'+str(y)])



        r_win_prob_test.append(radiant_match_winrate/5)

        d_win_prob_test.append(dire_match_winrate/5)



    # features

    X_heroes_train['r_win_prob'] = pd.Series(r_win_prob, index=X_heroes_train.index)

    X_heroes_train['d_win_prob'] = pd.Series(d_win_prob, index=X_heroes_train.index)



    X_heroes_test['r_win_prob'] = pd.Series(r_win_prob_test, index=X_heroes_test.index)

    X_heroes_test['d_win_prob'] = pd.Series(d_win_prob_test, index=X_heroes_test.index)



    X_heroes_train['win_prob'] = X_heroes_train['r_win_prob'] - X_heroes_train['d_win_prob']

    X_heroes_test['win_prob'] = X_heroes_test['r_win_prob'] - X_heroes_test['d_win_prob']

    return X_heroes_train, X_heroes_test
%%time

X_heroes_train, X_heroes_test = win_rates(X_heroes_train, X_heroes_test,df_train_targets)
X_heroes_train.head()
X_heroes_test.head()
X_heroes_train.to_csv('bag_of_heroes_and_win_prob_train.csv',index=False)

X_heroes_test.to_csv('bag_of_heroes_and_win_prob_test.csv',index=False)
train_visual = X_heroes_train[['r_win_prob','d_win_prob','win_prob']].copy()

train_visual['radiant_win'] = y_train.reset_index(drop=True)
plt.hist(train_visual.loc[train_visual.radiant_win==1, 'r_win_prob'].values, bins=16, density=True, alpha=0.4,color='green',label='Radiant');

plt.hist(train_visual.loc[train_visual.radiant_win==0, 'r_win_prob'].values, bins=16, density=True, alpha=0.4,color='red',label='Dire');

plt.legend()

plt.title('Distribution of wins by r_win_prob');
plt.hist(train_visual.loc[train_visual.radiant_win==1, 'win_prob'].values, bins=16,  density=True, alpha=0.4,color='green', label='Radiant');

plt.hist(train_visual.loc[train_visual.radiant_win==0, 'win_prob'].values, bins=16, density=True, alpha=0.4,color='red', label='Dire');

plt.legend()

plt.title('Distribution of wins by win_prob');
X_heroes_train_flip = X_heroes_train.copy()
def train_boh_flip(X_heroes_train_flip, X_heroes_train):

    players = [f'f{i}' for i in range(0, 115)] # r1, r2...

    for player in players:

        X_heroes_train_flip[player] = X_heroes_train[player].map({1: -1, -1: 1, 0:0})

    return X_heroes_train_flip
X_heroes_train_flip = train_boh_flip(X_heroes_train_flip, X_heroes_train)
def rd_hero_win(X_heroes_train_flip,X_heroes_train):

    X_heroes_train_flip['r_win_prob'] = X_heroes_train['d_win_prob']

    X_heroes_train_flip['d_win_prob'] = X_heroes_train['r_win_prob']

    X_heroes_train_flip['win_prob'] = X_heroes_train_flip['r_win_prob'] - X_heroes_train_flip['d_win_prob']

    return X_heroes_train_flip
X_heroes_train_flip = rd_hero_win(X_heroes_train_flip,X_heroes_train)
X_heroes_train_flip.index = np.arange(39675, 39675 + len(X_heroes_train_flip))

X_heroes_train_flip.to_csv('bag_of_heroes_and_win_prob_train_flip.csv',index=False)
X_heroes_train_flip.head()