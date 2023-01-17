import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('../input')
# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import implicit # implicit feedback recommendation library

import matplotlib.pyplot as plt # visualization

import seaborn as sns # wrapper to plt for ease of use

import time # timing

import scipy.sparse as sparse # sparse matrix support

import pickle # Python object serialization



from zipfile import ZipFile # ZIP file I/O

from IPython.display import display # dataframe rendering, etc.

from pathlib import Path



color = sns.color_palette()

sns.set_style('white')

%matplotlib inline
ds_dir = '../input/instacart-market-basket-analysis'



with ZipFile(os.path.join(ds_dir,"aisles.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"departments.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"order_products__prior.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"order_products__train.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"orders.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"products.csv.zip"), 'r') as zipObj:

    zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"sample_submission.csv.zip"), 'r') as zipObj:

    zipObj.extractall()
order_products_prior_df = pd.read_csv('order_products__prior.csv')

order_products_train_df = pd.read_csv('order_products__train.csv')

orders_df               = pd.read_csv('orders.csv')

products_df             = pd.read_csv('products.csv')
# Merge orders and products

order_products_full_df = pd.concat([order_products_prior_df, order_products_train_df])

merged_order_products_df = pd.merge(order_products_full_df, products_df, on='product_id', how='left')
def get_user_products_df(path, orders_df, order_products_df):

    '''

    Generates a dataframe of users and their product purchases, and writes it to disk at the given path

    '''

    start = time.time()

    print('Creating user-product dataframe... ', end='')

    

    # Consider any "prior" orders and remove all columns except `user_id` from `df_orders`

    order_user_df = orders_df[['order_id', 'user_id']]

    

    # Remove all columns except order_id and user_id from orders_df and merge the above on `order_id` and remove `order_id`

    merged_df = pd.merge(order_products_df, order_user_df, on='order_id').drop('order_id',axis=1)

    reordered_user_products_df = merged_df.groupby(['user_id', 'product_id']).reordered.sum()

    user_products_df = pd.merge(merged_df, reordered_user_products_df, how='left', on=['user_id', 'product_id']).drop(['reordered_x', 'add_to_cart_order'], axis=1)

    

    # Write to disk

    user_products_df.to_csv(path, index_label=False)

    

    print(f'Completed in {round(time.time() - start, 2)}s')



# Build dataframe of users and their product purchases (Needed for building the utility matrix)

REBUILD_MATRIX_DF_FULL = False

matrix_df_full_path = 'user_products_full.csv'

if REBUILD_MATRIX_DF_FULL or not Path(matrix_df_full_path).is_file():

    get_user_products_df(matrix_df_full_path, orders_df, order_products_full_df)



user_products_df = pd.read_csv(matrix_df_full_path)

user_products_df['user_id'] = user_products_df['user_id'].astype('category')

user_products_df['product_id'] = user_products_df['product_id'].astype('category')
def build_product_user_matrix_full(path, user_products_df):

    '''

    Generates a utility matrix representing purchase history of users, and writes it to disk.

    Rows and columns represent products and users respectively.

    '''

    start = time.time()

    print('Creating product-user matrix... ', end='')

    

    product_user_matrix = sparse.coo_matrix((user_products_df['reordered_y'],

                                            (user_products_df['product_id'].cat.codes.copy(),

                                             user_products_df['user_id'].cat.codes.copy())))

    sparse.save_npz(path, product_user_matrix)

    

    print(f'Completed in {round(time.time() - start, 2)}s')



REBUILD_MATRIX_FULL = False

matrix_full_path = 'product_user_matrix.npz'

if REBUILD_MATRIX_FULL or not Path(matrix_full_path).is_file():

    build_product_user_matrix_full(matrix_full_path, user_products_df)



product_user_matrix_full = sparse.load_npz(matrix_full_path).tocsr()
# How sparse is the utility matrix?

def sparsity(matrix):

    '''

    Given a matrix, returns its sparsity

    '''

    total_size = matrix.shape[0] * matrix.shape[1]

    actual_size = matrix.size

    sparsity = (1 - (actual_size / total_size)) * 100

    return sparsity



sparsity(product_user_matrix_full)
def confidence_matrix(product_user_matrix, alpha):

    '''

    Given a utility matrix, returns the given matrix converted to a confidence matrix

    (refer to http://yifanhu.net/PUB/cf.pdf for more details)

    '''

    return (product_user_matrix * alpha).astype('double')



def build_bpr(product_user_matrix, **kwargs):

    '''

    Given the utility matrix and model parameters,

    builds models and writes it to disk at a given path

    '''

    start = time.time()

    

    # Build model

    print(f'Building BPR model... ', end='')

    model = implicit.bpr.BayesianPersonalizedRanking()

    model.approximate_similar_items = False

    

    model.fit(product_user_matrix)

    

    # Save model to disk

    with open(kwargs['path'], 'wb+') as f:

        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    

    print(f'Completed in {round(time.time() - start, 2)}s')



# Specify model params and build it

bpr_params = {'random_state': 0}

bpr_params['path'] = 'imf_benchmark_bpr.pkl'



REBUILD_MODEL = True

if REBUILD_MODEL or not Path(bpr_params['path']).exists():

    build_bpr(product_user_matrix_full, **bpr_params)

with open(bpr_params['path'], 'rb') as f:

    bpr_model = pickle.load(f)
# Since the utility matrix is 0-indexed, the below dict is required to convert between `ids` and `indices`.

# For example, `product_id` 1 in the dataset is represented by the `0`th row of the utility matrix.



# Maps user_id: user index

u_dict = {uid : i for i, uid in enumerate(user_products_df['user_id'].cat.categories)}



# Maps product_index: product id

p_dict = dict(enumerate(user_products_df['product_id'].cat.categories))
orders_test_df = orders_df[orders_df.eval_set == 'test'][['user_id']]

relation_df = user_products_df[['user_id', 'product_id']]

relation_df.drop_duplicates(inplace=True)
sparse_user_product_matrix = product_user_matrix_full.T.tocsr()

N_REC = 100
def assign_recommendations(row):

    # print(f'Progress: {round((row.name + 1) * 100 / end, 2)}%...', end='\r', flush=True)

    return [(pid, score, rank) for rank, (pid, score) in enumerate(bpr_model.recommend(u_dict[row.user_id], sparse_user_product_matrix, N=N_REC), start=1)]
results_df = orders_test_df.reset_index(drop='index')

print('Recommending items... ', end='')

start = time.time()

results_df['products'] = results_df.apply(assign_recommendations, axis=1)

print(f'Completed in {round(time.time() - start, 2)}s')
results_df_new = results_df.explode('products').reset_index(drop='index').rename(columns={'products': 'product_id'})

results_df_new[['product_id', 'score', 'rank']] = pd.DataFrame(results_df_new['product_id'].tolist(), index=results_df_new.index)

results_df_new['product_id'] = results_df_new['product_id'].map(p_dict)

hasil = pd.merge(results_df_new, relation_df, how='inner', left_on=['user_id','product_id'], right_on=['user_id','product_id'])

hasil
def clean_prediction(row):

    data = row.products

    data = str("".join(str(data))[1:-1].replace(',',' '))

    return data
r_hasil = hasil.groupby('user_id')['product_id'].apply(list).reset_index(name='products')

r_hasil['products'] = r_hasil.apply(clean_prediction, axis=1)

r_hasil
submission_df = orders_df[orders_df.eval_set == 'test']

submission_df = submission_df[['order_id','user_id']]



sub_hasil = pd.merge(submission_df, r_hasil, how='outer', on='user_id').sort_values('user_id')

sub_hasil.fillna('None', inplace=True)

sub_hasil.drop('user_id', axis=1, inplace=True)

sub_hasil.to_csv('submission.csv', index=False)

sub_hasil
sub_hasil[sub_hasil['products'] != 'None']