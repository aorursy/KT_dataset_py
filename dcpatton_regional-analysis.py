import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 500)

import warnings  

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/rendimiento-escolar-chile/20180214_Resumen_Rendimiento 2017_20180131.csv'

                           , delimiter=';')

df.sample(10)
df.info(max_cols=500)
df.drop(['AGNO'], axis=1, inplace=True)
df.drop(['NOM_COM_RBD', 'COD_COM_RBD', 'NOM_DEPROV_RBD', 'COD_DEPROV_RBD'], axis=1, inplace=True)

df.drop(['COD_PRO_RBD'], axis=1, inplace=True) # removing province information
df.drop(['RURAL_RBD'], axis=1, inplace=True)
df.drop(['RBD', 'DGV_RBD', 'NOM_RBD'], axis=1, inplace=True)
df = df.drop(['APR_HOM_01','APR_HOM_02','APR_HOM_03','APR_HOM_04','APR_HOM_05','APR_HOM_06','APR_HOM_07','APR_HOM_08'

                  ,'APR_MUJ_01','APR_MUJ_02','APR_MUJ_03','APR_MUJ_04','APR_MUJ_05','APR_MUJ_06','APR_MUJ_07','APR_MUJ_08'

                  ,'APR_SI_03','APR_SI_04','APR_SI_07','RET_SI_01'

                  ,'REP_HOM_01','REP_HOM_02','REP_HOM_03','REP_HOM_04','REP_HOM_05','REP_HOM_06','REP_HOM_07','REP_HOM_08'

                  ,'REP_MUJ_01','REP_MUJ_02','REP_MUJ_03','REP_MUJ_04','REP_MUJ_05','REP_MUJ_06','REP_MUJ_07','REP_MUJ_08'

                  ,'RET_HOM_01','RET_HOM_02','RET_HOM_03','RET_HOM_04','RET_HOM_05','RET_HOM_06','RET_HOM_07','RET_HOM_08'

                  ,'RET_MUJ_01','RET_MUJ_02','RET_MUJ_03','RET_MUJ_04','RET_MUJ_05','RET_MUJ_06','RET_MUJ_07','RET_MUJ_08'

                  ,'TRA_HOM_01','TRA_HOM_02','TRA_HOM_03','TRA_HOM_04','TRA_HOM_05','TRA_HOM_06','TRA_HOM_07','TRA_HOM_08'

                  ,'TRA_MUJ_01','TRA_MUJ_02','TRA_MUJ_03','TRA_MUJ_04','TRA_MUJ_05','TRA_MUJ_06','TRA_MUJ_07','TRA_MUJ_08'

                  ,'SI_HOM_01','SI_HOM_02','SI_HOM_03','SI_HOM_04','SI_HOM_05','SI_HOM_07'

                  ,'SI_MUJ_01','SI_MUJ_02','SI_MUJ_03','SI_MUJ_04','SI_MUJ_05','SI_MUJ_07'], axis=1)
df.drop(['TRA_HOM_TO', 'TRA_MUJ_TO', 'APR_SI_TO', 'SI_HOM_TO', 'SI_MUJ_TO'], axis=1, inplace=True)
df['HOM_TO'] = df['APR_HOM_TO'] + df['REP_HOM_TO'] + df['RET_HOM_TO']

df['MUJ_TO'] = df['APR_MUJ_TO'] + df['REP_MUJ_TO'] + df['RET_MUJ_TO']



df['APR_HOM_TO'] = 100*(df['APR_HOM_TO']/df['HOM_TO'])

df['APR_MUJ_TO'] = 100*(df['APR_MUJ_TO']/df['MUJ_TO'])

df['REP_HOM_TO'] = 100*(df['REP_HOM_TO']/df['HOM_TO'])

df['REP_MUJ_TO'] = 100*(df['REP_MUJ_TO']/df['MUJ_TO'])

df['RET_HOM_TO'] = 100*(df['RET_HOM_TO']/df['HOM_TO'])

df['RET_MUJ_TO'] = 100*(df['RET_MUJ_TO']/df['MUJ_TO'])
def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)





encode_text_dummy(df, 'COD_REG_RBD') # this is categorical, not numeric



df = df.rename(columns={"COD_REG_RBD-1": "Tarapacá", "COD_REG_RBD-2": "Antofagasta"

                        ,"COD_REG_RBD-3": "Atacama", "COD_REG_RBD-4": "Coquimbo"

                        ,"COD_REG_RBD-5": "Valparaíso", "COD_REG_RBD-6": "O’Higgins"

                        ,"COD_REG_RBD-7": "Maule", "COD_REG_RBD-8": "Biobío"

                        ,"COD_REG_RBD-9": "Araucanía", "COD_REG_RBD-10": "Los Lagos"

                        ,"COD_REG_RBD-11": "Ibáñez", "COD_REG_RBD-12": "Antártica"

                        ,"COD_REG_RBD-13": "Santiago", "COD_REG_RBD-14": "Los Ríos"

                        ,"COD_REG_RBD-15": "Arica"})
# convert the categorical values into numeric types

from sklearn.preprocessing import LabelEncoder



for column in df.columns:

    if df[column].dtype == np.object:

        encoded = LabelEncoder()

        encoded.fit(df[column])

        df[column] = encoded.transform(df[column])
df.sample(5)
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



corr = df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
df.drop(['ESTADO_ESTAB'], axis=1, inplace=True)
df.drop(['PROM_ASIS_APR_HOM', 'PROM_ASIS_APR_MUJ', 'PROM_ASIS_APR_SI', 'PROM_ASIS_APR', 'PROM_ASIS_REP_HOM', 'PROM_ASIS_REP_MUJ',

       'PROM_ASIS_REP', 'PROM_ASIS'], axis=1, inplace=True) # missing data, percentages as string dtype
# is there more missing data

df.isnull().sum(axis=0)
df.dropna(axis=0, inplace=True) # remove rows with missing data
corr = df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
df.drop(['COD_ENSE', 'COD_ENSE2', 'COD_DEPE', 'COD_DEPE2'], axis=1, inplace=True)

df.sample(10)
corr = df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
df = pd.read_csv('/kaggle/input/rendimiento-escolar-chile/20180214_Resumen_Rendimiento 2017_20180131.csv'

                           , delimiter=';')

print(df.shape)

df = df.loc[df['COD_DEPE2'] ==1] # Municipales

print(df.shape)
df.drop(['AGNO'], axis=1, inplace=True)

df.drop(['NOM_COM_RBD', 'COD_COM_RBD', 'NOM_DEPROV_RBD', 'COD_DEPROV_RBD'], axis=1, inplace=True)

df.drop(['COD_PRO_RBD'], axis=1, inplace=True)

df.drop(['RURAL_RBD'], axis=1, inplace=True)

df.drop(['RBD', 'DGV_RBD', 'NOM_RBD'], axis=1, inplace=True)

df = df.drop(['APR_HOM_01','APR_HOM_02','APR_HOM_03','APR_HOM_04','APR_HOM_05','APR_HOM_06','APR_HOM_07','APR_HOM_08'

                  ,'APR_MUJ_01','APR_MUJ_02','APR_MUJ_03','APR_MUJ_04','APR_MUJ_05','APR_MUJ_06','APR_MUJ_07','APR_MUJ_08'

                  ,'APR_SI_03','APR_SI_04','APR_SI_07','RET_SI_01'

                  ,'REP_HOM_01','REP_HOM_02','REP_HOM_03','REP_HOM_04','REP_HOM_05','REP_HOM_06','REP_HOM_07','REP_HOM_08'

                  ,'REP_MUJ_01','REP_MUJ_02','REP_MUJ_03','REP_MUJ_04','REP_MUJ_05','REP_MUJ_06','REP_MUJ_07','REP_MUJ_08'

                  ,'RET_HOM_01','RET_HOM_02','RET_HOM_03','RET_HOM_04','RET_HOM_05','RET_HOM_06','RET_HOM_07','RET_HOM_08'

                  ,'RET_MUJ_01','RET_MUJ_02','RET_MUJ_03','RET_MUJ_04','RET_MUJ_05','RET_MUJ_06','RET_MUJ_07','RET_MUJ_08'

                  ,'TRA_HOM_01','TRA_HOM_02','TRA_HOM_03','TRA_HOM_04','TRA_HOM_05','TRA_HOM_06','TRA_HOM_07','TRA_HOM_08'

                  ,'TRA_MUJ_01','TRA_MUJ_02','TRA_MUJ_03','TRA_MUJ_04','TRA_MUJ_05','TRA_MUJ_06','TRA_MUJ_07','TRA_MUJ_08'

                  ,'SI_HOM_01','SI_HOM_02','SI_HOM_03','SI_HOM_04','SI_HOM_05','SI_HOM_07'

                  ,'SI_MUJ_01','SI_MUJ_02','SI_MUJ_03','SI_MUJ_04','SI_MUJ_05','SI_MUJ_07'], axis=1)

df.drop(['TRA_HOM_TO', 'TRA_MUJ_TO', 'APR_SI_TO', 'SI_HOM_TO', 'SI_MUJ_TO'], axis=1, inplace=True)

df.drop(['ESTADO_ESTAB'], axis=1, inplace=True)

df.drop(['PROM_ASIS_APR_HOM', 'PROM_ASIS_APR_MUJ', 'PROM_ASIS_APR_SI', 'PROM_ASIS_APR', 'PROM_ASIS_REP_HOM', 'PROM_ASIS_REP_MUJ',

       'PROM_ASIS_REP', 'PROM_ASIS'], axis=1, inplace=True) # missing data, percentages as string dtype

df.drop(['COD_ENSE', 'COD_ENSE2', 'COD_DEPE', 'COD_DEPE2'], axis=1, inplace=True)
df['HOM_TO'] = df['APR_HOM_TO'] + df['REP_HOM_TO'] + df['RET_HOM_TO']

df['MUJ_TO'] = df['APR_MUJ_TO'] + df['REP_MUJ_TO'] + df['RET_MUJ_TO']



df['APR_HOM_TO'] = 100*(df['APR_HOM_TO']/df['HOM_TO'])

df['APR_MUJ_TO'] = 100*(df['APR_MUJ_TO']/df['MUJ_TO'])

df['REP_HOM_TO'] = 100*(df['REP_HOM_TO']/df['HOM_TO'])

df['REP_MUJ_TO'] = 100*(df['REP_MUJ_TO']/df['MUJ_TO'])

df['RET_HOM_TO'] = 100*(df['RET_HOM_TO']/df['HOM_TO'])

df['RET_MUJ_TO'] = 100*(df['RET_MUJ_TO']/df['MUJ_TO'])
encode_text_dummy(df, 'COD_REG_RBD') # this is categorical, not numeric



df = df.rename(columns={"COD_REG_RBD-1": "Tarapacá", "COD_REG_RBD-2": "Antofagasta"

                        ,"COD_REG_RBD-3": "Atacama", "COD_REG_RBD-4": "Coquimbo"

                        ,"COD_REG_RBD-5": "Valparaíso", "COD_REG_RBD-6": "O’Higgins"

                        ,"COD_REG_RBD-7": "Maule", "COD_REG_RBD-8": "Biobío"

                        ,"COD_REG_RBD-9": "Araucanía", "COD_REG_RBD-10": "Los Lagos"

                        ,"COD_REG_RBD-11": "Ibáñez", "COD_REG_RBD-12": "Antártica"

                        ,"COD_REG_RBD-13": "Santiago", "COD_REG_RBD-14": "Los Ríos"

                        ,"COD_REG_RBD-15": "Arica"})
# convert the categorical values into numeric types

from sklearn.preprocessing import LabelEncoder



for column in df.columns:

    if df[column].dtype == np.object:

        encoded = LabelEncoder()

        encoded.fit(df[column])

        df[column] = encoded.transform(df[column])
df.dropna(axis=0, inplace=True) # remove rows with missing data

print(df.shape)
corr = df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)