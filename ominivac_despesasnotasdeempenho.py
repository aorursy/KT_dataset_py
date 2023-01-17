# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandas import read_csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)



df = read_csv('../input/DespesasNotasDeEmpenho-Copia.CSV', sep=';',low_memory= False,

              dtype={'isn_despesa_pesquisa_01': object, 'cod_ne': object,'cod_credor_cpf_cnpj':object,

                    'num_ano': object, 'cod_orgao': object, 'cod_secretaria': object, 'cod_credor': object,

                     'cod_fonte': object, 'cod_fonte_grupo': object, 'cod_funcao': object, 'cod_item': object,

                     'dth_liquidacao': object,

                     'cod_item_elemento': object, 'cod_item_categoria': object, 'cod_item_grupo': object,

                     'cod_item_modalidade': object, 'cod_licitacao_modalidade': object, 'cod_projeto_atividade': object,

                     'cod_poder': object, 'cod_programa': object , 'cod_subfuncao': object , 'num_sic': object,

                     'cod_np': object , 'cod_credor_cpf_cnpj': object, 'cod_org_externo':object ,

                     'cod_credor_pag':object , 'cod_subfonte':object , 'cod_destinacao':object 

                    }

             )

print(df)



    



df.dtypes
#df['isn_despesa_pesquisa_01'] = df['isn_despesa_pesquisa_01'].astype(str)
#df['num_ano'] = df['num_ano'].astype(str)
#df['cod_ne'] = df['cod_ne'].astype(str)
#df['cod_credor_cpf_cnpj'] = df['cod_credor_cpf_cnpj'].astype(str)
#df['dth_empenho'] = df['dth_empenho'].astype('datetime64[ns]')
#df['dth_liquidacao'] = df['dth_liquidacao'].astype('datetime64[ns]')
#df['dth_pagamento'] = df['dth_pagamento'].astype('datetime64[ns]')
df.dtypes
df.head(3)
grouped = df['cod_ne'].groupby([df['cod_ne'], df['dsc_secretaria']])
grouped
df.groupby("cod_ne")['dsc_secretaria'].apply(lambda tags: ','.join(tags))
g1 = df.groupby(['cod_ne','dsc_secretaria'])[['cod_ne','dsc_secretaria']].count()

print(g1)

g2 = df.groupby(['cod_ne','dsc_nome_credor'])[['cod_ne','dsc_nome_credor']].count()

print(g2)
df[df['dsc_nome_credor'].str.contains('FRANCISCO GILDEON FERNANDES NOBRE')]
serie_dsc_item_modalidade = df['dsc_item_modalidade'].value_counts()

serie_dsc_item_modalidade
serie_dsc_orgao =  df['dsc_orgao'].value_counts()

serie_dsc_orgao
df[df['dsc_nome_credor'] == 'TORINO INFORMATICA LTDA']
df[df['dsc_nome_credor'].str.contains('TORINO INFORMATICA ')]
df.loc[df.cod_ne== '20781']
df.loc[df.cod_ne == '15676']
groupby_regiment = df['cod_ne'].groupby(df['cod_ne'])

df.groupby('cod_ne').groups
grouped = df.groupby('cod_ne')

for name,group in grouped:

    print (name)

    print (group)
grouped = df['cod_ne'].groupby([df['cod_ne'], df['dsc_secretaria']])