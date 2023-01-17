
import numpy as np
import pandas as pd


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns

import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]].isnull().sum()
train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]]
train_transaction.head()
test_transaction.head()
train_transaction.columns[55:394]
test_transaction.columns[54:393]
test_transaction.iloc[:,55:393]
train_transaction.iloc[:,55:394]
train_transaction.iloc[:,55:393].corr()
train_transaction.iloc[:,55:393].isnull()
train_transaction.iloc[:,55:393].isnull().sum()
test_transaction.iloc[:,55:393].isnull().sum()
train_transaction.iloc[:,55:393].info()
train_transaction.iloc[:,55:393].describe().T
train_transaction.iloc[:,55:393].dtypes
test_transaction['isFraud']=2
transaction=pd.concat([train_transaction, test_transaction], ignore_index=True)
a=train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]]
b=test_transaction.loc[:,test_transaction.columns[test_transaction.columns.str.startswith('V')]]
del train_transaction
del test_transaction
transaction1=pd.concat([a, b], ignore_index=True)
def memory(df, verbose=True):
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


del transaction
del a
del b
data=memory(transaction1)
def PCA_(df, cols, prefix='PCA_', rand_seed=4):
    pca = PCA(random_state=rand_seed)
    pca.fit_transform(df[cols])
    represent=np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
    print(represent)
    n_components=0
    for i in represent:
        
        n_components+=1
        if i >=98:
            print("n_components= ",n_components)
            break
            
    pca = PCA(random_state=rand_seed,n_components=n_components)
    principalComponents = pca.fit_transform(df[cols])
    
    principalDf = pd.DataFrame(principalComponents)

    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, principalDf], axis=1)
    print(pca.explained_variance_ratio_)
    return df
V_columns= transaction1.columns[55:]

for col in V_columns:
    data[col] = data[col].fillna((data[col].min() - 1))
    data[col] = (minmax_scale(data[col], feature_range=(0,1)))
data=PCA_(data,V_columns,prefix='PCA_V_')