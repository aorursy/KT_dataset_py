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
train_users_game1 = pd.read_csv('/kaggle/input/ds2019uec-task2/train_users_game1.csv')

train_users_game2 = pd.read_csv('/kaggle/input/ds2019uec-task2/train_users_game2.csv')

test_users_game1 = pd.read_csv('/kaggle/input/ds2019uec-task2/test_users_game1.csv')

test_user_ids = pd.read_csv('/kaggle/input/ds2019uec-task2/test_user_ids.csv')

game_group2 = pd.read_csv('/kaggle/input/ds2019uec-task2/game_group2.csv')

sample_submission = pd.read_csv('/kaggle/input/ds2019uec-task2/sample_submission.csv')
users_game1 = pd.concat([train_users_game1, test_users_game1]).drop(['play_hour'], axis=1).drop_duplicates()

users_game1['label'] = 1
users_game1
user_ids = pd.concat([train_users_game1['user_id'], train_users_game2['user_id'], test_user_ids['user_id']]).unique()
users_game1_matrix = users_game1.set_index(['user_id', 'game_title'])['label'].unstack().reindex(user_ids).fillna(0)
users_game1_matrix
train_user_ids = pd.concat([train_users_game1['user_id'], train_users_game2['user_id']]).unique()

test_user_ids = test_user_ids['user_id'].values
from sklearn.metrics.pairwise import cosine_distances

users_similarity = 1 - cosine_distances(users_game1_matrix.loc[test_user_ids], users_game1_matrix.loc[train_user_ids])
users_similarity.shape
users_game2 = train_users_game2.drop(['play_hour', 'predict_game_id'], axis=1).drop_duplicates()

users_game2['label'] = 1
users_game2
users_game2_matrix = users_game2.set_index(['user_id', 'game_title'])['label'].unstack().reindex(user_ids).fillna(0)

users_game2_matrix.columns = users_game2_matrix.columns.map(game_group2.set_index('game_title')['predict_game_id'])
users_game2_matrix
test_users_game2_matrix = (users_similarity @ users_game2_matrix.loc[train_user_ids])
test_users_game2_matrix
sample_submission['purchased_games'] = test_users_game2_matrix.apply(lambda x: ' '.join(x.sort_values(ascending=False)[:10].index.astype('str')), axis=1)
sample_submission.loc[test_users_game2_matrix.sum(axis=1) == 0, 'purchased_games'] = ' '.join(train_users_game2['predict_game_id'].value_counts(normalize=True).index.astype('str')[:10])
sample_submission.to_csv('submission.csv', index=None)