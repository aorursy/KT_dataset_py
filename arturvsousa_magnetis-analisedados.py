print("Estas duas linhas são código em Python. A de cima é o comando feito, e a de baixo é o que retorna desse comando.")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams.update({'font.size': 15})

plt.rcParams['figure.figsize'] = 18, 5

pd.set_option('display.max_colwidth', 50)
users = pd.read_csv('../input/navegacao-usuarios/navegacao_usuarios.csv')
users.head()
users.tail()
users.info()
18243-5834
users.loc[users['Valor Simulado'].isnull(), 'Valor Simulado'] = 0
users['Valor Simulado'].value_counts()
users.loc[users['Página'] == '1 - Homepage', 'Página'] = 1
users.loc[users['Página'] == '2 - Questionário', 'Página'] = 2
users.loc[users['Página'] == '3 - Plano de Investimentos', 'Página'] = 3
users.loc[users['Página'] == '4 - Cadastro', 'Página'] = 4
users['Página'].unique()
users['Página'] = users['Página'].astype(int)
users.info()
ax = sns.countplot(x="Página", data=users)

ax.set_ylabel('Número de usuários')

ax.patches[0].set_color('#6c84bf')

ax.patches[1].set_color('#d29a57')

ax.patches[2].set_color('#81cc78')

ax.patches[3].set_color('#cd4729')
plt.figure(figsize=(20,8))

geral_pizza = (users['Página'].value_counts(normalize=True)*100).plot(kind='pie', autopct='%1.1f%%', colors=(['#6c84bf', '#d29a57', '#81cc78', '#cd4729']))

geral_pizza.set_ylabel('')
avancaram = users[users['Id'].duplicated(keep=False)]
avancaram = avancaram.sort_values(['Id', 'Página'])
avancaram = avancaram.drop_duplicates(subset='Id', keep='last')
avancaram.head()
nao_avancaram = users.drop_duplicates(subset='Id', keep=False)
nao_avancaram
funil_ok = pd.concat([avancaram, nao_avancaram])
funil_ok = funil_ok.sort_values(['Id'])
funil_ok.head()
ax = sns.countplot(x="Página", data=funil_ok)

ax.set_ylabel('Número de usuários')

ax.patches[0].set_color('#6c84bf')

ax.patches[1].set_color('#d29a57')

ax.patches[2].set_color('#81cc78')

ax.patches[3].set_color('#cd4729')

    
plt.figure(figsize=(20,8))

funil_pizza = (funil_ok['Página'].value_counts(normalize=True)*100).plot(kind='pie', autopct='%1.1f%%', colors=(['#6c84bf', '#d29a57', '#cd4729', '#81cc78']))

funil_pizza.set_ylabel('')
funil_ok[funil_ok['Página'] == 4]['Valor Simulado'].value_counts().head(15)
(funil_ok[funil_ok['Página'] == 4]['Valor Simulado'].value_counts(normalize=True)*100).head(15)
funil_ok[funil_ok['Página'] == 4]['Valor Simulado'].describe()
funil_ok[funil_ok['Página'] == 2]['Origem de Tréfego'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 2]['Dispositivo'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 4]['Origem de Tréfego'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 4]['Dispositivo'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 1]['Origem de Tréfego'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 1]['Dispositivo'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 3]['Origem de Tréfego'].value_counts(normalize=True)*100
funil_ok[funil_ok['Página'] == 3]['Dispositivo'].value_counts(normalize=True)*100
ads = funil_ok[funil_ok['Origem de Tréfego'] == 'Paid Search']

non_ads = funil_ok[funil_ok['Origem de Tréfego'] != 'Paid Search']
ads_convert = len(ads[ads['Página'] == 4])
n_ads_conv = len(non_ads[non_ads['Página'] == 4])
ads_convert
n_ads_conv
ads_chi_sq = ((212-785.5)**2)/785.5

ads_chi_sq
nads_chi_sq = ((1359-785.5)**2)/785.5

nads_chi_sq
grupos_chi_sq = ads_chi_sq + nads_chi_sq

grupos_chi_sq
chi_squared_values = []



for i in range(1000):

    seq = np.random.random((1571,))

    seq[seq < 0.5] = 0

    seq[seq >= 0.5] = 1

    ads_count = len(seq[seq == 0])

    nads_count = len(seq[seq == 1])

    ads_diff = ((ads_count - 785.5)**2)/785.5

    nads_diff = ((nads_count - 785.5)**2)/785.5

    chi_sq = ads_diff + nads_diff

    chi_squared_values.append(chi_sq)
plt.hist(chi_squared_values)
max(chi_squared_values)
plt.figure(figsize=(20,8))

device_pizza = (funil_ok[funil_ok['Página'] == 4]['Dispositivo'].value_counts(normalize=True)*100).plot(kind='pie', autopct='%1.1f%%')

device_pizza.set_ylabel('')
funil_valores = funil_ok[funil_ok['Valor Simulado'] > 0.0]

funil_valores = funil_valores[['Dispositivo' , 'Valor Simulado']]

funil_valores.groupby(by='Dispositivo').describe().plot(kind='bar', ylim=(0,60000), rot=0).set_ylabel('Valor Simulado')

funil_ok['Origem de Tréfego'].value_counts().plot(kind='bar', rot=0)
funil_ok[funil_ok['Página'] == 4]['Origem de Tréfego'].value_counts().plot(kind='bar', rot=0)