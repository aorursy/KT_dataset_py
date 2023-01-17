from sklearn import preprocessing



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('/kaggle/input/b3-stock-quotes/COTAHIST_A2009_to_A2020_P.csv', 

                 parse_dates=['DATPRE', 'DATVEN'],

                 date_parser=lambda x: pd.to_datetime(x, errors='coerce'),

                 index_col=0)
df.head()
df.columns.values
df.dtypes
df.describe()
df.drop(columns=['TIPREG', 'MODREF', 'INDOPC'], inplace=True)
df = df[(df.PRAZOT.isna()) | (df.PRAZOT == 0)]
df = df[df.CODNEG.str.len() == 5]
df_bdi = pd.get_dummies(df['CODBDI'], prefix='BDI')

df.drop(columns=['CODBDI'], inplace=True)

df = pd.concat([df, df_bdi], axis=1)
df_market_type = pd.get_dummies(df['TPMERC'], prefix='TPMERC')

df.drop(columns=['TPMERC'], inplace=True)

df = pd.concat([df, df_market_type], axis=1)
cols = ['PREABE', 'PREMAX', 'PREMIN', 'PREMED', 'PREULT', 'PREOFC', 'PREOFV']



for col in cols:

    df[col + '_fat'] = df[col] / df['FATCOT']
df.head()
df.columns.values
df.dtypes
df.describe()
def group_quotes(min_date=None, max_date=None, min_year=None, max_year=None):

    if min_date != None and max_date != None:

        df_base = df[(df['DATPRE'] >= min_date) & (df['DATPRE'] <= max_date)]

    elif max_date != None:

        df_base = df[(df['DATPRE'] <= max_date)]

    elif min_date != None:

        df_base = df[(df['DATPRE'] >= min_date)]

    else:

        df_base = df

    

    df_cod_min_max_dt = df_base.groupby(by=['CODNEG'], as_index=False).agg({'DATPRE': ['min', 'max']})

    df_cod_min_max_dt.columns = ['CODNEG', 'DATPRE_MIN', 'DATPRE_MAX']

    df_cod_min_max_dt['PREMED_MIN'] = pd.merge(df_cod_min_max_dt.rename(columns={'DATPRE_MIN': 'DATPRE'}), df_base, how='inner', on=['CODNEG', 'DATPRE'])['PREMED']

    df_cod_min_max_dt['PREMED_MAX'] = pd.merge(df_cod_min_max_dt.rename(columns={'DATPRE_MAX': 'DATPRE'}), df_base, how='inner', on=['CODNEG', 'DATPRE'])['PREMED']

    

    if min_year != None and max_year != None:

        df_cod_min_max_dt = df_cod_min_max_dt[(df_cod_min_max_dt['DATPRE_MIN'].dt.year == min_year) & (df_cod_min_max_dt['DATPRE_MAX'].dt.year == max_year)]

    elif max_year != None:

        df_cod_min_max_dt = df_cod_min_max_dt[(df_cod_min_max_dt['DATPRE_MAX'].dt.year == max_year)]

    elif min_year != None:

        df_cod_min_max_dt = df_cod_min_max_dt[(df_cod_min_max_dt['DATPRE_MIN'].dt.year == min_year)]

        

    return df_cod_min_max_dt
df_cod_min_max_dt = group_quotes(min_date='2015-01-01', max_year=2020)

df_cod_min_max_dt['INC'] = df_cod_min_max_dt['PREMED_MAX'] / df_cod_min_max_dt['PREMED_MIN']

df_cod_min_max_dt['INC'] = preprocessing.MinMaxScaler().fit_transform(np.reshape(df_cod_min_max_dt['INC'].values, (-1, 1)))

df_cod_min_max_dt.sort_values(by='INC', ascending=False, inplace=True)



fig = plt.figure()

ax = fig.add_axes([0,0,2,1])

best_quotes = df_cod_min_max_dt[:30]

ax.bar(best_quotes['CODNEG'], best_quotes['INC'])

ax.set_xticklabels(best_quotes['CODNEG'], rotation=45)

plt.show()
def plot_quote_timeserie(code):

    quote = df[df['CODNEG'] == code][['PREMED', 'DATPRE']]

    quote.set_index('DATPRE', inplace=True)

    fig = plt.figure()

    ax = fig.add_axes([0,0,2,1])

    ax.plot(quote)

    plt.show()
plot_quote_timeserie('BBDC4')