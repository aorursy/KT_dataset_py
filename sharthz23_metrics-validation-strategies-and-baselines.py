import os

import numpy as np 

import pandas as pd 

from itertools import islice, cycle

from more_itertools import pairwise



print('Dataset:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/mts-library/interactions.csv')

df_users = pd.read_csv('../input/mts-library/users.csv')

df_items = pd.read_csv('../input/mts-library/items.csv')
df.info()
df['start_date'] = pd.to_datetime(df['start_date'])
duplicates = df.duplicated(subset=['user_id', 'item_id'], keep=False)

df_duplicates = df[duplicates].sort_values(by=['user_id', 'start_date'])

df = df[~duplicates]
df_duplicates = df_duplicates.groupby(['user_id', 'item_id']).agg({

    'progress': 'max',

    'rating': 'max',

    'start_date': 'min'

})

df = df.append(df_duplicates.reset_index(), ignore_index=True)
df['progress'] = df['progress'].astype(np.int8)

df['rating'] = df['rating'].astype(pd.SparseDtype(np.float32, np.nan))
df.info()
df.to_pickle('interactions_preprocessed.pickle')
!ls -lah
df_users.head()
df_users.info()
df_users.nunique()
df_users['age'] = df_users['age'].astype('category')

df_users['sex'] = df_users['sex'].astype(pd.SparseDtype(np.float32, np.nan))
df_users.info()
interaction_users = df['user_id'].unique()



common_users = len(np.intersect1d(interaction_users, df_users['user_id']))

users_only_in_interaction = len(np.setdiff1d(interaction_users, df_users['user_id']))

users_only_features = len(np.setdiff1d(df_users['user_id'], interaction_users))

total_users = common_users + users_only_in_interaction + users_only_features

print(f'Кол-во пользователей - {total_users}')

print(f'Кол-во пользователей c взаимодействиями и фичами - {common_users} ({common_users / total_users * 100:.2f}%)')

print(f'Кол-во пользователей только c взаимодействиями - {users_only_in_interaction} ({users_only_in_interaction / total_users * 100:.2f}%)')

print(f'Кол-во пользователей только c фичами - {users_only_features} ({users_only_features / total_users * 100:.2f}%)')
df_users.to_pickle('users_preprocessed.pickle')
!ls -lah
df_items.head()
df_items.info(memory_usage='full')
def num_bytes_format(num_bytes, float_prec=4):

    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb']

    for unit in units[:-1]:

        if abs(num_bytes) < 1000:

            return f'{num_bytes:.{float_prec}f} {unit}'

        num_bytes /= 1000

    return f'{num_bytes:.4f} {units[-1]}'
num_bytes = df_items.memory_usage(deep=True).sum()

num_bytes_format(num_bytes)
df_items.nunique()
df_items['year'].value_counts().tail(25)
df_items[df_items['year'] == '1898, 1897, 1901']
for col in ['genres', 'authors', 'year']:

    df_items[col] = df_items[col].astype('category')
df_items.info(memory_usage='full')
num_bytes = df_items.memory_usage(deep=True).sum()

num_bytes_format(num_bytes)
interaction_items = df['item_id'].unique()



common_items = len(np.intersect1d(interaction_items, df_items['id']))

items_only_in_interaction = len(np.setdiff1d(interaction_items, df_items['id']))

items_only_features = len(np.setdiff1d(df_items['id'], interaction_items))

total_items = common_items + items_only_in_interaction + items_only_features

print(f'Кол-во книг - {total_items}')

print(f'Кол-во книг c взаимодействиями и фичами - {common_items} ({common_items / total_items * 100:.2f}%)')

print(f'Кол-во книг только c взаимодействиями - {items_only_in_interaction} ({items_only_in_interaction / total_items * 100:.2f}%)')

print(f'Кол-во книг только c фичами - {items_only_features} ({items_only_features / total_items * 100:.2f}%)')
df_items.to_pickle('items_preprocessed.pickle')
!ls -lah
df_true = pd.DataFrame({

    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],

    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],

    'value':   [4,                    5,                    3,            5]

})

df_true
df_recs = pd.DataFrame({

    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],

    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],

    'value':   [3.28,                 3.5,                  4.06,           4.73]

})

df_recs
df_true = df_true.set_index(['user_id', 'item_id'])

df_recs = df_recs.set_index(['user_id', 'item_id'])



df_merged = df_true.join(df_recs, how='left', lsuffix='_true', rsuffix='_recs')

df_merged
df_merged['MAE'] = (df_merged['value_true'] - df_merged['value_recs']).abs()

df_merged['MSE'] = (df_merged['value_true'] - df_merged['value_recs']) ** 2

df_merged
print(f"MAE  - {df_merged['MAE'].mean():.4f}")

print(f"MSE  - {df_merged['MSE'].mean():.4f}")

print(f"RMSE - {np.sqrt(df_merged['MSE'].mean()):.4f}")
df_true = pd.DataFrame({

    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],

    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],

})

df_true
df_recs = pd.DataFrame({

    'user_id': [

        'Аня', 'Аня', 'Аня', 

        'Боря', 'Боря', 'Боря', 

        'Вася', 'Вася', 'Вася',

    ],

    'item_id': [

        'Отверженные', 'Двенадцать стульев', 'Герои нашего времени', 

        '451° по Фаренгейту', '1984', 'О дивный новый мир',

        'Десять негритят', 'Искра жизни', 'Зеленая миля', 

    ],

    'rank': [

        1, 2, 3,

        1, 2, 3,

        1, 2, 3,

    ]

})

df_recs
df_merged = df_true.set_index(['user_id', 'item_id']).join(df_recs.set_index(['user_id', 'item_id']), how='left')

df_merged
df_merged['hit@2'] = df_merged['rank'] <= 2

df_merged
df_merged['hit@2/2'] = df_merged['hit@2'] / 2

df_merged
df_prec2 = df_merged.groupby(level=0)['hit@2/2'].sum()

df_prec2
print(f'Precision@2 - {df_prec2.mean()}')
df_merged['hit@2/2'].sum() / df_merged.index.get_level_values('user_id').nunique()
users_count = df_merged.index.get_level_values('user_id').nunique()

for k in [1, 2, 3]:

    hit_k = f'hit@{k}'

    df_merged[hit_k] = df_merged['rank'] <= k

    print(f'Precision@{k} = {(df_merged[hit_k] / k).sum() / users_count:.4f}')
df_merged['users_item_count'] = df_merged.groupby(level='user_id')['rank'].transform(np.size)

df_merged
for k in [1, 2, 3]:

    hit_k = f'hit@{k}'

    # Уже посчитано

    # df_merged[hit_k] = df_merged['rank'] <= k  

    print(f"Recall@{k} = {(df_merged[hit_k] / df_merged['users_item_count']).sum() / users_count:.4f}")
df_true = pd.DataFrame({

    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],

    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],

})

df_true 
df_recs = pd.DataFrame({

    'user_id': [

        'Аня', 'Аня', 'Аня', 

        'Боря', 'Боря', 'Боря', 

        'Вася', 'Вася', 'Вася',

    ],

    'item_id': [

        'Отверженные', 'Двенадцать стульев', 'Герои нашего времени', 

        '451° по Фаренгейту', '1984', 'О дивный новый мир',

        'Десять негритят', 'Рита Хейуорт и спасение из Шоушенка', 'Зеленая миля', 

    ],

    'rank': [

        1, 2, 3,

        1, 2, 3,

        1, 2, 3,

    ]

})

df_recs
df_merged = df_true.set_index(['user_id', 'item_id']).join(df_recs.set_index(['user_id', 'item_id']), how='left')

df_merged = df_merged.sort_values(by=['user_id', 'rank'])

df_merged
df_merged['reciprocal_rank'] = 1 / df_merged['rank']

df_merged
mrr = df_merged.groupby(level='user_id')['reciprocal_rank'].max()

mrr
print(f"MRR = {mrr.fillna(0).mean()}")
df_merged['cumulative_rank'] = df_merged.groupby(level='user_id').cumcount() + 1

df_merged['cumulative_rank'] = df_merged['cumulative_rank'] / df_merged['rank']

df_merged['users_item_count'] = df_merged.groupby(level='user_id')['rank'].transform(np.size)

df_merged
users_count = df_merged.index.get_level_values('user_id').nunique()

map3 = (df_merged["cumulative_rank"] / df_merged["users_item_count"]).sum() / users_count

print(f"MAP@3 = {map3}")
test_dates = df['start_date'].unique()[-7:]

test_dates
test_dates = list(pairwise(test_dates))

test_dates
split_dates = test_dates[0]

train = df[df['start_date'] < split_dates[0]]

test = df[(df['start_date'] >= split_dates[0]) & (df['start_date'] < split_dates[1])]

test = test[(test['rating'] >= 4) | (test['rating'].isnull())]

split_dates, train.shape, test.shape
class PopularRecommender():

    def __init__(self, max_K=100, days=30, item_column='item_id', dt_column='date'):

        self.max_K = max_K

        self.days = days

        self.item_column = item_column

        self.dt_column = dt_column

        self.recommendations = []

        

    def fit(self, df, ):

        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)

        self.recommendations = df.loc[df[self.dt_column] > min_date, self.item_column].value_counts().head(self.max_K).index.values

    

    def recommend(self, users=None, N=10):

        recs = self.recommendations[:N]

        if users is None:

            return recs

        else:

            return list(islice(cycle([recs]), len(users)))
pop_model = PopularRecommender(days=7, dt_column='start_date')

pop_model.fit(train)
top10_recs = pop_model.recommend()

top10_recs
item_titles = pd.Series(df_items['title'].values, index=df_items['id']).to_dict()

item_titles[128115]
list(map(item_titles.get, top10_recs))
recs = pd.DataFrame({'user_id': test['user_id'].unique()})

top_N = 10

recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)

recs.head()
recs = recs.explode('item_id')

recs.head(top_N + 2)
recs['rank'] = recs.groupby('user_id').cumcount() + 1

recs.head(top_N + 2)
test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))

test_recs = test_recs.sort_values(by=['user_id', 'rank'])

test_recs.tail()
test_recs['users_item_count'] = test_recs.groupby(level='user_id', sort=False)['rank'].transform(np.size)

test_recs['reciprocal_rank'] = 1 / test_recs['rank']

test_recs['reciprocal_rank'] = test_recs['reciprocal_rank'].fillna(0)

test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1

test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']

test_recs.tail()
test_recs[test_recs['rank'].notnull()].head()
print(f'Метрик по test ({str(split_dates[0])[:10]}, {str(split_dates[1])[:10]})')

users_count = test_recs.index.get_level_values('user_id').nunique()

for k in range(1, top_N + 1):

    hit_k = f'hit@{k}'

    test_recs[hit_k] = test_recs['rank'] <= k

    print(f'Precision@{k} = {(test_recs[hit_k] / k).sum() / users_count:.4f}')

    print(f"Recall@{k} = {(test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count:.4f}")



mapN = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count

print(f"MAP@{top_N} = {mapN}")



mrr = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()

print(f"MRR = {mrr}")