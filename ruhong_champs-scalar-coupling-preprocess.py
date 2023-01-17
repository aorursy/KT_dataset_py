# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm



# Input data files are available in the "../input/" directory.



# Any results you write to the current directory are saved as output.
SEED = 31
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
pd.set_option('display.max_columns', 500)
file_folder = '../input/champs-scalar-coupling/'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

sub = pd.read_csv(f'{file_folder}/sample_submission.csv')

structures = pd.read_csv(f'{file_folder}/structures.csv')
print(f'There are {train.shape[0]} rows in train data.')

print(f'There are {test.shape[0]} rows in test data.')



print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")

print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")

print(f"There are {structures['atom'].nunique()} unique atoms.")

print(f"There are {train['type'].nunique()} unique types.")
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df





def make_features(df):

    df = map_atom_info(df, 0)

    df = map_atom_info(df, 1)

    df_p_0 = df[['x_0', 'y_0', 'z_0']].values

    df_p_1 = df[['x_1', 'y_1', 'z_1']].values

    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)

    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2

    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2

    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2

    df['type_0'] = df['type'].apply(lambda x: x[0])

    df['type_1'] = df['type'].apply(lambda x: x[1:])

    df['dist_to_type_mean'] = df['dist'] / df.groupby('type')['dist'].transform('mean')

    df['dist_to_type_0_mean'] = df['dist'] / df.groupby('type_0')['dist'].transform('mean')

    df['dist_to_type_1_mean'] = df['dist'] / df.groupby('type_1')['dist'].transform('mean')

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')

    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']

    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = df.fillna(0)

    return df

%%time



train = make_features(train)

test = make_features(test)

%%time



def encode(df, col, encoder):

    df[col] = df[col].map(encoder).fillna(0)

    assert df[col].isnull().sum() == 0



def freq_encode(col):

    encoder = dict(train[col].value_counts(normalize=True))

    encode(train, col, encoder)

    encode(test, col, encoder)





freq_encode('type')

freq_encode('type_0')

freq_encode('type_1')

freq_encode('atom_0')

freq_encode('atom_1')
def log_transform(df, col, offset=1):

    df[col] = np.log(df[col] + offset)



def hist_npp_charts(df, col):

    """histogram and normal probability plot"""

    sns.distplot(df[col], fit=norm);

    fig = plt.figure()

    res = stats.probplot(df[col], plot=plt)

    
hist_npp_charts(train, 'scalar_coupling_constant')
#log_transform(train, 'scalar_coupling_constant', offset=1000)

#hist_npp_charts(train, 'scalar_coupling_constant')
train.head()
#correlation matrix

corrmat = train.corr()

k = 80 #number of variables for heatmap

cols = corrmat.nlargest(k, 'scalar_coupling_constant')['scalar_coupling_constant'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

plt.subplots(figsize=(40, 40))

sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# Unused because I wasn't sure of its exact behaviour.

def corr_columns(df, threshold=0.97):

    print("Starting Preprocessing..")

    # Absolute value correlation matrix

    corr_matrix = df.corr().abs()

    # Getting the upper triangle of correlations

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop

    
keep_columns = [

    'scalar_coupling_constant',

    'id',

    'molecule_atom_index_0_dist_mean_div',

    'molecule_atom_index_0_dist_max_div',

    'molecule_atom_index_1_dist_max_div',

    'molecule_atom_index_0_dist_std_div',

    'molecule_atom_index_0_dist_min_div',

    'molecule_atom_index_1_dist_mean_div',

    'molecule_atom_index_1_dist_std_div',

    'molecule_atom_1_dist_std_diff',

    'molecule_atom_index_0_dist_std_diff',

    'molecule_atom_index_0_dist_mean_diff',

    'molecule_atom_index_1_dist_max_diff',

    'molecule_atom_index_0_dist_max_diff',

    'molecule_type_0_dist_std_diff',

    'molecule_atom_index_1_dist_mean_diff',

    'molecule_atom_index_1_dist_std_diff',

    'molecule_atom_1_dist_min_div',

    'molecule_atom_1_dist_min_diff',

    'type',

    'type_0',

    'type_1',

    'molecule_type_dist_min',

    'molecule_type_dist_mean',

    'molecule_type_0_dist_std',

    'dist_to_type_1_mean',

    'dist',

    'molecule_type_dist_max',

    'dist_x',

    'dist_y',

    'dist_z'

]

train = train[keep_columns]

test = test[keep_columns[1:]]

print(f'len(keep_columns)={len(keep_columns)}, train={train.shape}, test={test.shape}')
%%time



print(f'train={train.shape}, test={test.shape}')

train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)

print(os.listdir("."))