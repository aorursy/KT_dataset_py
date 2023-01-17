import numpy as np

import pandas as pd 

import os, glob
!wget -q -r ftp://ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/IPCA_15/Resultados_por_Subitem/2017/

!wget -q -r ftp://ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/IPCA_15/Resultados_por_Subitem/2018/

!mkdir cache

!for z in ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/IPCA_15/Resultados_por_Subitem/2017/*.zip; do unzip -qq "$z" -d cache; done

!for z in ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/IPCA_15/Resultados_por_Subitem/2018/*.zip; do unzip -qq "$z" -d cache; done

!rm -r ftp.ibge.gov.br
filenames = glob.glob('./cache/*.xls')
ipca15 = pd.read_excel(filenames[0], index_col=[0], header=[4], skiprows=[5])
# Select some rows

cols_cat = [' ÍNDICE GERAL',       # 'GENERAL INDEX'

        ' ALIMENTAÇÃO E BEBIDAS',     # 'FOOD AND BEVERAGES'

        ' HABITAÇÃO',                 # 'HOUSING',

        ' IOGURTE E BEBIDAS LÁCTEAS', # 'YOGURT AND DAIRY BEVERAGES',

        ' ARTIGOS DE RESIDÊNCIA',     # 'RESIDENCE ARTICLES',

        ' VESTUÁRIO',                 # 'CLOTHING',

        ' TRANSPORTES',               # 'TRANSPORTES',

        ' SAÚDE E CUIDADOS PESSOAIS', # 'HEALTH AND PERSONAL CARE',

        ' DESPESAS PESSOAIS',         # 'PERSONAL EXPENSES',

        ' EDUCAÇÃO',                  # 'EDUCATION',

        ' COMUNICAÇÃO',               # 'COMMUNICATION'

        ]
mth = filenames[0].split('_')[1][:6]

print(f"Month: {mth[:4]+'-'+mth[-2:]}")

# Cols meaning :

# Monthly variation by groups (%)

# Rio de Janeiro, Porto Alegre, Belo Horizonte, Recife,

# São Paulo, Brasília, Belém, Fortaleza, Salvador, Curitiba, Goiânia

ipca15.drop_duplicates().loc[cols_cat, :]
# And now collect colomn 'NACIONAL' in DataFrame by month

IPCA15 = pd.DataFrame(index=pd.DatetimeIndex(freq='M', start='2017-01-01', end='2018-12').to_period('M')

                          , columns=cols_cat)



for file in filenames:

    mth = file.split('_')[1][:6]

    idx = pd.to_datetime(mth[:4]+'-'+mth[-2:]).to_period('M')

    tmp = pd.read_excel(file, index_col=[0], header=[4], skiprows=[5])

    IPCA15.loc[idx,:] = tmp.drop_duplicates().loc[cols_cat, 'NACIONAL'].values

IPCA15[cols_cat] = IPCA15[cols_cat].astype(np.float32)

IPCA15.index.name='eval month'
IPCA15.head()
# read the data

df = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])

trns = pd.read_csv('../input/historical_transactions.csv',

                   parse_dates=['purchase_date'], infer_datetime_format=True)
cols = ['card_id', 'month_lag', 'purchase_date']

df = pd.merge(df, trns[cols].groupby('card_id').first(), on='card_id', left_index=True)
df['eval month'] = df.purchase_date - df.month_lag.astype('timedelta64[M]')

df['eval month'] = df['eval month'].dt.to_period('M')

df.drop(['month_lag', 'purchase_date'], axis=1, inplace=True)
df.head()
df_stats = pd.merge(df, IPCA15.reset_index(), on='eval month', left_index=True)
# check correlation with target, it's 99% coincidence, but mb usefull

df_stats.corr().iloc[4:,3:4]
IPCA15.to_csv('IPCA15.csv')
!rm -r cache

!wget -q -r -l 1 ftp://ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/INPC/Resultados_por_Subitem/ -A zip

!wget -q -r ftp://ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/INPC/Resultados_por_Subitem/2017    

!mkdir cache

!for z in ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/INPC/Resultados_por_Subitem/2017/*.zip; do unzip -qq "$z" -d cache; done

!for z in ftp.ibge.gov.br/Precos_Indices_de_Precos_ao_Consumidor/INPC/Resultados_por_Subitem/*.zip; do unzip -qq "$z" -d cache; done

!rm -r ftp.ibge.gov.br
filenames = glob.glob('./cache/*.xls')

# And now collect colomn 'NACIONAL' in DataFrame by month

INPC = pd.DataFrame(index=pd.DatetimeIndex(freq='M', start='2017-01-01', end='2018-12').to_period('M')

                          , columns=cols_cat)



for file in filenames:

    mth = file.split('_')[1][:6]

    idx = pd.to_datetime(mth[:4]+'-'+mth[-2:]).to_period('M')

    tmp = pd.read_excel(file, index_col=[0], header=[4], skiprows=[5])

    INPC.loc[idx,:] = tmp.drop_duplicates().loc[cols_cat, 'NACIONAL'].values

INPC[cols_cat] = INPC[cols_cat].astype(np.float32)

INPC.index.name='eval month'
df_stats = pd.merge(df, INPC.reset_index(), on='eval month', left_index=True)
df_stats.corr().iloc[4:,3:4]
INPC.to_csv('INPC.csv')
!rm -r cache