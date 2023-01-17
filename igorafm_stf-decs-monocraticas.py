import numpy as np 

import pandas as pd 

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style





%matplotlib inline

print("Bibliotecas carregadas")
stf = pd.read_csv('../input/STF_dec_mono_texto.csv', sep='|')

print('Dados carregados')
stf
stf.shape
stf.loc[0, 'Decisao']
stf["Classe_abrev"].value_counts()
stf[stf['Classe_abrev'] == 'AORCPR732']
sns.set_context("poster")

style.use('fivethirtyeight')

fig, ax = plt.subplots()

fig.set_size_inches(19.27, 30.7)

ax = sns.countplot(y="Classe_abrev", data=stf, palette=('Set1'))

ax.set_title('Classes processuais presentes na base de dados')

plt.figtext(0.95, 0.05, 'Fonte: Legal Hackers Natal', horizontalalignment='right')

ax.set(ylabel='Classes (abrevida)')

ax.set(xlabel='Quantidade')

figura = ax.get_figure()
from datetime import datetime





stf['Data_Decisao'] = stf['Data_Decisao'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))

stf
stf.sort_values(by='Data_Decisao', ascending=False)
stf[stf['Relator'] == 'celso de mello']
stf2 = pd.read_csv('../input/STF_dec_mono_class.csv', sep='|', low_memory=False)

print("Dados carregados")
stf2
stf2.shape
stf2['Andamento'].value_counts()
stf2['Assuntos']
stf2.loc[99, 'Assuntos']
stf3 = pd.read_csv('../input/STF_dec_mono_texto_class.csv', sep='|')

stf3
for item in stf3.loc[14, ['Assuntos', 'Decisao']]:

    print(item)