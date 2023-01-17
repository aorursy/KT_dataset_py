import numpy as np 

import pandas as pd 

import scipy.sparse as sp



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/mts-library/interactions.csv')

df.head()
df['start_date'] = pd.to_datetime(df['start_date'])
df.info()
duplicates = df.duplicated(subset=['user_id', 'item_id'], keep=False)

duplicates.sum()
df_duplicates = df[duplicates].sort_values(by=['user_id', 'start_date'])

df = df[~duplicates]
df_duplicates = df_duplicates.groupby(['user_id', 'item_id']).agg({

    'progress': 'max',

    'rating': 'max',

    'start_date': 'min'

})

df_duplicates.info()
df = df.append(df_duplicates.reset_index(), ignore_index=True)

df.info()
df.nunique()
df_cat = pd.DataFrame({'city': ['Moscow', 'London', 'Tokyo', 'Moscow']})

df_cat
df_cat['city'] = df_cat['city'].astype('category')

df_cat
df_cat['city_codes'] = df_cat['city'].cat.codes

df_cat
mapping = dict(enumerate(df_cat['city'].cat.categories))

mapping
df_user_item = df[['user_id', 'item_id']].copy()
def num_bytes_format(num_bytes, float_prec=4):

    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb']

    for unit in units[:-1]:

        if abs(num_bytes) < 1000:

            return f'{num_bytes:.{float_prec}f} {unit}'

        num_bytes /= 1000

    return f'{num_bytes:.4f} {units[-1]}'
num_bytes_ints = df_user_item.memory_usage(deep=True).sum()

num_bytes_format(num_bytes_ints)
df_user_item = df_user_item.astype('string')

num_bytes_str = df_user_item.memory_usage(deep=True).sum()

num_bytes_format(num_bytes_str)
df_user_item = df_user_item.astype('category')

num_bytes_cat = df_user_item.memory_usage(deep=True).sum()

num_bytes_format(num_bytes_cat)
print(f'Экономия category относительно string: {(1 - num_bytes_cat / num_bytes_str) * 100:.2f}%')

print(f'Экономия ints относительно category: {(1 - num_bytes_ints / num_bytes_cat) * 100:.2f}%')
df_user_item = df_user_item.astype(np.int64).astype('category')

num_bytes_int_cat = df_user_item.memory_usage(deep=True).sum()

num_bytes_format(num_bytes_int_cat)
print(f'Экономия category on int64 относительно category on string: {(1 - num_bytes_int_cat / num_bytes_cat) * 100:.2f}%')
df_user_item['user_id'].cat.codes.dtype
ratings = df['rating'].astype(np.float32).copy()
num_bytes_float = ratings.memory_usage(deep=True)

num_bytes_format(num_bytes_float)
ratings = ratings.astype(pd.Int32Dtype())

num_bytes_Int32 = ratings.memory_usage(deep=True)

num_bytes_format(num_bytes_Int32)
ratings = ratings.astype(pd.Int8Dtype())

num_bytes_Int8 = ratings.memory_usage(deep=True)

num_bytes_format(num_bytes_Int8)
ratings
print(f'Экономия Int8DType относительно float64: {(1 - num_bytes_Int8 / num_bytes_float) * 100:.2f}%')
sparse_type = pd.SparseDtype(np.float32, np.nan)

ratings = ratings.astype(np.float32).astype(sparse_type)
ratings
num_bytes_sparse = ratings.memory_usage(deep=True)

num_bytes_format(num_bytes_sparse)
print(f'Экономия sparse относительно Int8DType: {(1 - num_bytes_sparse / num_bytes_Int8) * 100:.2f}%')

print(f'Экономия sparse относительно float32: {(1 - num_bytes_sparse / num_bytes_float) * 100:.2f}%')
ratings.sparse.density
rows =   [1,  1, 0,  4,   2, 2]

cols =   [0,  1, 0,  5,   3, 3]

values = [-2, 7, 19, 1.0, 6, 8]



coo = sp.coo_matrix((values, (rows, cols)))

coo
coo.todense()
coo.row, coo.col, coo.data
csr = coo.tocsr()

csr
csr.todense()
csr.indptr, csr.indices, csr.data
csc = coo.tocsc()

csc
csc.todense()
csc.indptr, csc.indices, csc.data
df.head()
df.info()
df.nunique()
users_inv_mapping = dict(enumerate(df['user_id'].unique()))

users_mapping = {v: k for k, v in users_inv_mapping.items()}

len(users_mapping)
users_mapping[126706], users_inv_mapping[0]
items_inv_mapping = dict(enumerate(df['item_id'].unique()))

items_mapping = {v: k for k, v in items_inv_mapping.items()}

len(items_mapping)
items_mapping[14433], items_inv_mapping[0]
rows = df['user_id'].map(users_mapping.get)

cols = df['item_id'].map(items_mapping.get)



rows.isna().sum(), cols.isna().sum()
coo = sp.coo_matrix((

    np.ones(df.shape[0], dtype=np.int8),

    (rows, cols)

))

coo
num_bytes_format(coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)
df['weight'] = ((df['progress'] + 1) / 101) * (2 ** df['rating'])

df['weight'] = df['weight'].astype(np.float32)
ax = df['weight'].plot.hist()
coo = sp.coo_matrix((

    df['weight'],

    (rows, cols)

))

coo
num_bytes_format(coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)