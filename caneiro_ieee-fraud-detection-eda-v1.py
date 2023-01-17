import numpy as np 

import pandas as pd 

import pandas_profiling as pp

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_colwidth', -1)



# Função reduce memory -> https://www.kaggle.com/cttsai

def reduce_mem_usage(df, verbose=True):

    start_mem = df.memory_usage().sum() / 1024**2

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



    for col in df.columns:

#         print(col)

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

        # RCO - Acrescentado para conversao de Yes or No em 1 e 0

        elif col_type == 'object':

            if df[col].unique().all() in ['Y', 'N', 'Yes', 'No', 'Sim', 'Não', 'Verdadeiro', 'Falso']:

                df[col] = df[col].map({'Y': 1, 'N': 0}).astype('int8')

                                

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



def import_merge(tipo, nr=None):

    trans = pd.read_csv(DATA_FOLDER + tipo + '_transaction.csv', nrows=nr)

    ident = pd.read_csv(DATA_FOLDER + tipo + '_identity.csv', nrows=nr)

    ident[id] = 1 # para identificação das linhas correspondentes no merge

    df = trans.merge(ident, how='left')

    df = reduce_mem_usage(df)

    return df



def describe_object(df):

    df1 = pd.DataFrame()

    for col in df.select_dtypes(include = 'object').columns: # somente colunas texto (categóricas)

        item = df[col].dropna().unique() # lista de valores unicos na coluna

        nulo = df[col].isna().sum() # quantidade te valores null

        pnulo = nulo / len(df[col].index) # % de valores null

        dic = {'ind': col,'lista':[item], 'nulos':nulo, '%nulo':pnulo} #monta um dict 

        df2 = pd.DataFrame(dic) # passa para df

        df1 = pd.concat([df1,df2]) # vai juntando cada coluna como um registro novo no df

    df1.set_index('ind', drop=True, inplace=True) # define index como nome da variavel

    df3 = df.describe(include = 'object').T

    df3['%freq'] = df3['freq'].div(df3['count']).astype(np.float64).round(4)  # inclui campo com % da moda

    df4 = df3.merge(df1, left_index = True, right_index = True) # merge c/ describe do pd

    return df4







DATA_FOLDER = '../input/'

import os

print(os.listdir(DATA_FOLDER))
df_train = import_merge('train', 10000)
describe_object(df_train)