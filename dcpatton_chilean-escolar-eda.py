import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 500)

import warnings  

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/rendimiento-escolar-chile/20180214_Resumen_Rendimiento 2017_20180131.csv'

                           , delimiter=';')

df.sample(10)
df.info(max_cols=500)
df.describe()
df.shape
# number of missing values



df.isnull().sum(axis=0)
df['NOM_COM_RBD'].value_counts() # number of schools in each community
# convert the categorical value into numeric type

from sklearn.preprocessing import LabelEncoder



for column in df.columns:

    if df[column].dtype == np.object:

        encoded = LabelEncoder()

        encoded.fit(df[column])

        df[column] = encoded.transform(df[column])
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



corr = df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
df_tot = df.drop(['APR_HOM_01','APR_HOM_02','APR_HOM_03','APR_HOM_04','APR_HOM_05','APR_HOM_06','APR_HOM_07','APR_HOM_08'

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
corr = df_tot.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
name = 'RURAL_RBD'

dummies = pd.get_dummies(df_tot[name])

for x in dummies.columns:

    dummy_name = f"{name}-{x}"

    df_tot[dummy_name] = dummies[x]

df_tot.drop(name, axis=1, inplace=True)

df_tot = df_tot.rename(columns={"RURAL_RBD-0": "Urban", "RURAL_RBD-1": "Rural"})



corr = df_tot.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)
df_tot['Pass_avg'] = (df_tot['APR_HOM_TO']+df_tot['APR_MUJ_TO'])/(df_tot['APR_HOM_TO']+df_tot['APR_MUJ_TO']

                                                                  +df_tot['REP_HOM_TO']+df_tot['REP_MUJ_TO']

                                                                 +df_tot['RET_HOM_TO']+df_tot['RET_MUJ_TO'])

corr = df_tot.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, annot=True)