import numpy as np 

import pandas as pd 

import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
rawdata = '/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv'

pd.options.display.float_format = '{:.2f}'.format

sns.set(style='whitegrid')
df = pd.read_csv(rawdata)
df.columns = [cols.replace("(","").replace(")","").replace(",","").replace("'","").replace(" ","_") for cols in df.columns]
df.head()
#seta novo index

df.set_index('P0_id', inplace=True)
#arrumar nome de coluna com erros de ortografia

cols = df.columns.tolist()

cols[47] = 'P22_most_used_programming_languages'

cols[160] = 'P35_data_science_platforms_preference'

df.columns = cols
#seleciona colunas que possuem valores null

null_cols = df.columns[df.isnull().any()].tolist()

df_null = df[null_cols]
#valores absolutos e percentuais de valores nulos por coluna

msno.bar(df_null,figsize=(20,8), color='darkturquoise',fontsize=14, labels=True);
#substituir os valores nulos por -1

df.loc[df['P1_age'].isnull(), 'P1_age'] = -1

#altera o datatype

df['P1_age'] = df['P1_age'].astype('int8')
df.loc[df['P2_gender'].isnull(), 'P2_gender'] = 'ausente/excluído'

df['P2_gender'] = df['P2_gender'].astype('category')
df.loc[df['P5_living_state'].isnull(), 'P5_living_state'] = 'ausente/excluído'

df['P5_living_state'] = df['P5_living_state'].astype('category')
df.loc[df['P6_born_or_graduated'].isnull(), 'P6_born_or_graduated'] = -1

df['P6_born_or_graduated'] = df['P6_born_or_graduated'].astype('int8')
df.loc[df['P12_workers_number'].isnull(), 'P12_workers_number'] = 'ausente/excluído'

df['P12_workers_number'] = df['P12_workers_number'].astype('category')
df.loc[df['P13_manager'].isnull(), 'P13_manager'] = -1

df['P13_manager'] = df['P13_manager'].astype('int8')
df.loc[df['P16_salary_range'].isnull(), 'P16_salary_range'] = 'ausente/excluído'

df['P16_salary_range'] = df['P16_salary_range'].astype('category')
df.loc[df['P22_most_used_programming_languages'].isnull(), 'P22_most_used_programming_languages'] = 'ausente/excluído'

df['P22_most_used_programming_languages'] = df['P22_most_used_programming_languages'].astype('category')
df.loc[df['P29_have_data_warehouse'].isnull(), 'P29_have_data_warehouse'] = -1

df['P29_have_data_warehouse'] = df['P29_have_data_warehouse'].astype('int8')
df.loc[df['P35_data_science_platforms_preference'].isnull(), 'P35_data_science_platforms_preference'] = 'ausente/excluído'

df['P35_data_science_platforms_preference'] = df['P35_data_science_platforms_preference'].astype('category')
df['P35_other'] = df['P35_other'].str.lower()

df['P35_other'] = df['P35_other'].apply(lambda x : x.strip() if type(x) == str else x)
mapping_others = {

    'data science academy' : 'Data Science Academy',

    'dsa' : 'Data Science Academy',

    'datascienceacademy' : 'Data Science Academy',

    'minerando dados' : 'Minerando Dados',

    'youtube' : 'YouTube',

    'linkedin learning' : 'LinkedIn Learning',

    'pluralsight' : 'Pluralsight',

    'data science academy' : 'Data Science Academy',

    'data sciences academy' : 'Data Science Academy', 

    'data science academy' : 'Data Science Academy',

    'datascience academy' : 'Data Science Academy',

    'cognitive class.ai' : 'cognitiveclass.ai',

    'cognitive ai' : 'cognitiveclass.ai',

    'sigmoidal' : 'sigmoidal.ai',

    'pluralsights' : 'Pluralsight',

    'blog minerando dados' : 'Minerando Dados',

    'dsa academy' : 'Data Science Academy',

    'data scienceacademy' : 'Data Science Academy',

    'data academy' : 'Data Science Academy',

}
df['P35_other'] = df['P35_other'].map(mapping_others).fillna(df['P35_other'])
#mantem apenas os top registros, essa coluna possui uma cauda longa. Portanto, faz sentido agrupar os registros com poucas observações como "outros"

for col in df.groupby('P35_other')['P1_age'].count().sort_values(ascending=False)[4:].index.tolist():

    mapping_others[col] = 'Outros'

mapping_others['Data Science Academy'] = 'Data Science Academy'

mapping_others['cognitiveclass.ai'] = 'cognitiveclass.ai'

mapping_others['Minerando Dados'] = 'Minerando Dados'
df['P35_other'] = df['P35_other'].map(mapping_others).fillna('Não Preenchido')

df['P35_other'] = df['P35_other'].astype('category')
df.loc[df['P36_draw_participation'].isnull(), 'P36_draw_participation'] = -1

df['P36_draw_participation'] = df['P36_draw_participation'].astype('int8')
df.loc[df['D1_living_macroregion'].isnull(), 'D1_living_macroregion'] = 'ausente/excluído'

df['D1_living_macroregion'] = df['D1_living_macroregion'].astype('category')
df.loc[df['D2_origin_macroregion'].isnull(), 'D2_origin_macroregion'] = 'ausente/excluído'

df['D2_origin_macroregion'] = df['D2_origin_macroregion'].astype('category')
df.loc[df['D3_anonymized_degree_area'].isnull(), 'D3_anonymized_degree_area'] = 'ausente/excluído'

df['D3_anonymized_degree_area'] = df['D3_anonymized_degree_area'].astype('category')
df.loc[df['D4_anonymized_market_sector'].isnull(), 'D4_anonymized_market_sector'] = 'ausente/excluído'

df['D4_anonymized_market_sector'] = df['D4_anonymized_market_sector'].astype('category')
df.loc[df['D5_anonymized_manager_level'].isnull(), 'D5_anonymized_manager_level'] = 'ausente/excluído'

df['D5_anonymized_manager_level'] = df['D5_anonymized_manager_level'].astype('category')
df.loc[df['D6_anonymized_role'].isnull(), 'D6_anonymized_role'] = 'ausente/excluído'

df['D6_anonymized_role'] = df['D6_anonymized_role'].astype('category')
#mapear o data type das colunas

container_for_types = {}

for colname, coltype in zip(df.dtypes.index, df.dtypes.values):

    col_datatype = coltype.type.__name__

    if col_datatype not in container_for_types:

        container_for_types[col_datatype] = []

    container_for_types[col_datatype].append(colname)
for col in container_for_types['object_']:

    df[col] = df[col].astype('category')
for col in container_for_types['int64']:

    df[col] = df[col].astype('int8')
#mapear o data type das colunas

container_for_types = {}

for colname, coltype in zip(df.dtypes.index, df.dtypes.values):

    col_datatype = coltype.type.__name__

    if col_datatype not in container_for_types:

        container_for_types[col_datatype] = []

    container_for_types[col_datatype].append(colname)
#salva um csv com os tipos das colunas do dataset

with open("DH_survey_dtypes.csv", "w") as file_:

    file_.write("col_name,col_dtype\n")

    for dtype, colname in container_for_types.items():

        if dtype == 'CategoricalDtypeType':

            dtype = 'category'

        for col in colname:

            file_.write(f"{col},{dtype}\n")
#salva arquivo com dados tratados

df.to_csv("pesquisa-data-hackers-2019-cleaned.csv")
df_dtypes = pd.read_csv("DH_survey_dtypes.csv").set_index('col_name')
dtypes_dict = df_dtypes.col_dtype.to_dict()
#load a cleaned dataset with correct dtypes

df_cleaned = pd.read_csv('pesquisa-data-hackers-2019-cleaned.csv', index_col='P0_id', dtype=dtypes_dict)
df_cleaned.dtypes[:15]
df_cleaned.head()