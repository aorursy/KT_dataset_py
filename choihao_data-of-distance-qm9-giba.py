%matplotlib inline



import pandas as pd

import numpy as np



import math

import gc

import copy

import os



from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt

import seaborn as sns



from lightgbm import LGBMRegressor
#设置数据输入路径和导出路径

DATA_PATH = '../input'

SUBMISSIONS_PATH = './'

#用原子序数代替原子名称

ATOMIC_NUMBERS = {

    'H': 1,

    'C': 6,

    'N': 7,

    'O': 8,

    'F': 9

}
#设置显示参数

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 120)

pd.set_option('display.max_columns', 120)
#显示当前路径下的文件夹

os.listdir(DATA_PATH)
#定义减小消耗内存函数

def reduce_mem_usage(df, verbose=False):

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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_csv = pd.read_csv('../input/read-giba/giba_train.csv')

test_csv = pd.read_csv('../input/read-giba/giba_test.csv')

train_csv.head(10)
test_csv.head(10)
#减小消耗内存，防止处理数据时内存不足

train_csv = reduce_mem_usage(train_csv,verbose = True)

test_csv = reduce_mem_usage(test_csv,verbose = True)
gc.collect()
#使用新的数值型特征molecule_index代替字符型molecule_name

train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

test_csv['molecule_index'] = test_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

#删除molecule_name列

train_csv = train_csv.drop(columns = 'molecule_name',axis = 1)

test_csv = test_csv.drop(columns = 'molecule_name',axis = 1)

train_csv.head(10)
test_csv.head(10)
submission_csv = pd.read_csv(f'{DATA_PATH}/champs-scalar-coupling/sample_submission.csv', index_col='id')
#设置QM9数据集使用的特征

qm9_columns = ['mulliken_min', 'mulliken_max', 'mulliken_atom_0', 'mulliken_atom_1']
print("Load QM9 features...")

data_qm9 = pd.read_pickle('../input/quantum-machine-9-qm9/data.covs.pickle')

data_qm9.head(10)
#删除不使用的特征列

data_qm9 = data_qm9.drop(columns = ['type', 'linear', 'atom_index_0', 'atom_index_1', 

                                    'scalar_coupling_constant', 'U', 'G', 'H', 

                                    'mulliken_mean', 'r2', 'U0','rc_A','rc_B',

                                    'rc_C', 'mu', 'alpha', 'homo','lumo', 'gap',

                                    'zpve', 'Cv', 'freqs_min', 'freqs_max', 'freqs_mean',], axis=1)

data_qm9 = reduce_mem_usage(data_qm9,verbose=True)
data_qm9.head(10)
#data_qm9.set_index('id',inplace = True)

#使用新的数值型特征molecule_index代替字符型molecule_name

data_qm9['molecule_index'] = data_qm9.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

#删除molecule_name列

data_qm9 = data_qm9.drop(columns = 'molecule_name',axis = 1)
data_qm9.head(10)
train_csv = pd.merge(train_csv, data_qm9, how='left', on=['molecule_index','id'])

test_csv = pd.merge(test_csv, data_qm9, how='left', on=['molecule_index','id'])

#删除data_qm9文件，防止内存不足

del data_qm9

gc.collect()
train_csv.set_index('id',inplace = True)

#train_csv = train_csv.drop(['molecule_name'],axis =1)

train_csv.head(10)
test_csv.set_index('id',inplace = True)

#test_csv = test_csv.drop(['molecule_name'],axis =1)

test_csv.head(10)
train_csv.columns
structures_dtypes = {

    'molecule_name': 'category',

    'atom_index': 'int8',

    'atom': 'category',

    'x': 'float32',

    'y': 'float32',

    'z': 'float32'

}

structures_csv = pd.read_csv(f'{DATA_PATH}/champs-scalar-coupling/structures.csv', dtype=structures_dtypes)

structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]

structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')

structures_csv.head(10)
print('Shape: ', structures_csv.shape)

print('Total: ', structures_csv.memory_usage().sum())

structures_csv.memory_usage()

structures_csv = reduce_mem_usage(structures_csv,verbose = True)
#将所有的缺失数据填充为0

train_csv = train_csv.fillna(0)

test_csv = test_csv.fillna(0)

structures_csv = structures_csv.fillna(0)
train_csv.to_csv(f'{SUBMISSIONS_PATH}/fin_train.csv')

test_csv.to_csv(f'{SUBMISSIONS_PATH}/fin_test.csv')

structures_csv.to_csv(f'{SUBMISSIONS_PATH}/fin_structures.csv')