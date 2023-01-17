# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

pd.set_option('display.max_columns', 200)

pd.set_option('display.width', 400)

pd.set_option('display.max_rows',600)

import matplotlib.pyplot as plt

import seaborn as sns
df_enem = pd.read_csv('/kaggle/input/enem-por-escola-2005-a-2015/microdados_enem_por_escola/DADOS/MICRODADOS_ENEM_ESCOLA.csv',sep=';',engine='python')

df_censo = pd.read_csv('/kaggle/input/sensoescolar/2015/DADOS/ESCOLAS/ESCOLAS.CSV',sep='|',engine='python')

df_enem.head()
df_censo.head()
df_enem_filtro = df_enem[df_enem['SG_UF_ESCOLA']=='DF']

df_enem_filtro = df_enem_filtro[df_enem_filtro['NU_ANO']==2015]

df_enem_filtro.head()
df_censo_filtro = df_censo[df_censo['CO_UF']== 53]

df_censo_filtro.head()
df_enem_filtro.describe()
df_censo_filtro.describe()
df_escola_biblioteca = df_censo_filtro[df_censo_filtro['IN_BIBLIOTECA']== 1]

df_escola_sem_biblioteca = df_censo_filtro[df_censo_filtro['IN_BIBLIOTECA']== 0]

df_biblioteca = df_escola_biblioteca[df_escola_biblioteca['CO_ENTIDADE'] != 0]

df_sem_biblioteca = df_escola_sem_biblioteca[df_escola_sem_biblioteca['CO_ENTIDADE']!=0]

df_enem_filtro.rename(columns={'CO_ESCOLA_EDUCACENSO': 'CO_ENTIDADE'}, inplace=True)

df_enem_biblioteca = pd.merge(df_enem_filtro, df_biblioteca, on=['CO_ENTIDADE'], how='inner')

df_enem_sem_biblioteca = pd.merge(df_enem_filtro, df_sem_biblioteca, on=['CO_ENTIDADE'], how='inner')

df_enem_biblioteca_publica = df_enem_biblioteca[df_enem_biblioteca['TP_DEPENDENCIA'] == 2]

df_enem_biblioteca_privada = df_enem_biblioteca[df_enem_biblioteca['TP_DEPENDENCIA'] == 4]

df_enem_sem_biblioteca_publica = df_enem_sem_biblioteca[df_enem_sem_biblioteca['TP_DEPENDENCIA'] == 2]

df_enem_sem_biblioteca_privada = df_enem_sem_biblioteca[df_enem_sem_biblioteca['TP_DEPENDENCIA'] == 4]



#sns.catplot(x="CO_ENTIDADE", y="NU_MEDIA_RED",kind="bar",data=df_enem_sem_biblioteca)



sns.catplot(x="CO_ENTIDADE", y="NU_MEDIA_RED",kind="bar",data=df_enem_biblioteca_publica)

print("Média de redeção escolas públicas com biblioteca: ",df_enem_biblioteca_publica["NU_MEDIA_RED"].mean())

media_publica_bb = df_enem_biblioteca_publica["NU_MEDIA_RED"].mean()
sns.catplot(x="CO_ENTIDADE", y="NU_MEDIA_RED",kind="bar",data=df_enem_biblioteca_privada)

print("Média de redeção escolas privadas com biblioteca: ",df_enem_biblioteca_privada["NU_MEDIA_RED"].mean())

media_privada_bb = df_enem_biblioteca_privada["NU_MEDIA_RED"].mean()
sns.catplot(x="CO_ENTIDADE", y="NU_MEDIA_RED",kind="bar",data=df_enem_sem_biblioteca_publica)

print("Média de redeção escolas públicas sem biblioteca: ",df_enem_sem_biblioteca_publica["NU_MEDIA_RED"].mean())

media_publica_sem_bb = df_enem_sem_biblioteca_publica["NU_MEDIA_RED"].mean()
sns.catplot(x="CO_ENTIDADE", y="NU_MEDIA_RED",kind="bar",data=df_enem_sem_biblioteca_privada)

print("Média de redeção escolas privadas sem biblioteca: ",df_enem_sem_biblioteca_privada["NU_MEDIA_RED"].mean())

media_privada_sem_bb = df_enem_sem_biblioteca_privada["NU_MEDIA_RED"].mean()
labels = ['Privada', 'Pública']

biblioteca = [media_privada_bb, media_publica_bb]

sem_biblioteca = [media_privada_sem_bb, media_publica_sem_bb]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2,biblioteca, width, label='Com biblioteca')

rects2 = ax.bar(x + width/2, sem_biblioteca, width, label='Sem biblioteca')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Media da redação')

ax.set_title('Comparação de medias da redação em relação a ter ou não ter biblioteca')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



plt.show()
X = df_enem_biblioteca.drop('IN_REFEITORIO',axis=1)

y = df_enem_biblioteca['IN_REFEITORIO']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)
X.dtypes.sample(191)
X.select_dtypes(include='object')