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
train_pda2019_df = pd.read_csv('../input/pda2019/train-PDA2019.csv')



# Drop timestamp columns and convert integer ratings to real numbers

train_pda2019_df.drop('timeStamp', inplace=True, axis=1)

train_pda2019_df['userID'] = train_pda2019_df['userID'].astype('category')

train_pda2019_df['itemID'] = train_pda2019_df['itemID'].astype('category')

train_pda2019_df['rating'] = train_pda2019_df['rating'].astype('double')



display(train_pda2019_df.head())

print('Empty fields:')

print(train_pda2019_df.isna().sum())
def build_item_user_matrix(path, user_item_df):

    start = time.time()

    print('Creating product-user matrix... ', end='')

    

    item_user_matrix = sparse.coo_matrix((user_item_df['rating'],

                                         (user_item_df['itemID'].cat.codes.copy(),

                                          user_item_df['userID'].cat.codes.copy())))

    sparse.save_npz(path, item_user_matrix)

    

    print(f'Completed in {round(time.time() - start, 2)}s')



REBUILD_MATRIX_FULL = False

matrix_path = 'item_user_matrix.npz'

if REBUILD_MATRIX_FULL or not Path(matrix_path).is_file():

    build_item_user_matrix(matrix_path, train_pda2019_df)



item_user_matrix = sparse.load_npz(matrix_path).tocsr()
# How sparse is the utility matrix?

def sparsity(matrix):

    '''

    Given a matrix, returns its sparsity

    '''

    total_size = matrix.shape[0] * matrix.shape[1]

    actual_size = matrix.size

    sparsity = (1 - (actual_size / total_size)) * 100

    return sparsity



sparsity(item_user_matrix)
# Specify model params and build it

models = {

    'ALS': implicit.als.AlternatingLeastSquares,

    'BPR': implicit.bpr.BayesianPersonalizedRanking,

    'LMF': implicit.lmf.LogisticMatrixFactorization,

}



trained_models = dict()



params = {

    'ALS': {'random_state': 0},

    'BPR': {'random_state': 0},

    'LMF': {'random_state': 0},

}



for model_name, model_params in params.items():

    model_params['path'] = f'pda2019_imf_benchmark_{model_name.lower()}.pkl'



params
REBUILD_MODEL = True



def build_model(item_user_matrix):

    for name in models:

        if REBUILD_MODEL or not Path(params[name]['path']).exists():

            start = time.time()



            print(f'Building {name} model... ', end='')

            current_model = models[name]()



            current_model.fit(item_user_matrix)



            # Save model to disk

            with open(params[name]['path'], 'wb+') as f:

                pickle.dump(current_model, f, pickle.HIGHEST_PROTOCOL)

                print(f'Model saved to {params[name]["path"]}')



            print(f'Completed in {round(time.time() - start, 2)}s')



build_model(item_user_matrix)



for model_name, model_params in params.items():

    with open(model_params['path'], 'rb') as f:

        trained_models[model_name] = pickle.load(f)
# Maps user_id: user index

u_dict = {uid : i for i, uid in enumerate(train_pda2019_df['userID'].cat.categories)}



# Maps item_index: item id

i_dict = dict(enumerate(train_pda2019_df['itemID'].cat.categories))



sparse_user_item_matrix = item_user_matrix.T.tocsr()
test_pda2019_df = pd.read_csv('../input/pda2019/test-PDA2019.csv')

N_REC = 10



for name, model in trained_models.items():

    submission_df = test_pda2019_df.copy(deep=True)

    submission_df['recommended_itemIDs'] = submission_df['userID'].apply(lambda x: ' '.join([str(i_dict[rec[0]]) for rec in model.recommend(u_dict[x], sparse_user_item_matrix, N=N_REC)]))

    submission_df.to_csv(f'submission-PDA2019-{name.lower()}.csv', index=False)

    display(submission_df.head())