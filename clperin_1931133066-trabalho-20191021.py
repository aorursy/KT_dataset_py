# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


atletas = pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

regioes = pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')



print('Atletas:', atletas.shape)

print('Regiões:', regioes.shape)
regioes.info()
region_nulos = regioes[regioes['region'].isnull()]
region_nulos.sample(3).T
regioes[regioes['NOC'] == 'ROT']
regioes.drop(168, inplace=True)
regioes[regioes['NOC'] == 'TUV']
regioes.drop(208, inplace=True)
regioes[regioes['NOC'] == 'UNK']
regioes.drop(213, inplace=True)
region_nulos.rename(columns={'notes': 'regionc'}, inplace=True)
region_nulos.rename(columns={'region': 'notes'}, inplace=True)
region_nulos.rename(columns={'regionc': 'region'}, inplace=True)

region_nulos.sample(3)
regioes.drop('notes', axis=1, inplace=True)
regioes = regioes.append(region_nulos, ignore_index=True)
regioes[regioes['NOC'] == 'ROT']
regioes[regioes['NOC'] == 'UNK']
regioes[regioes['NOC'] == 'TUV']

atletas.sample(5).T
regioes.sample(5).T
atletasreg = pd.merge(atletas[atletas["Season"] != 'Winter'], regioes, how='left', on=['NOC', 'NOC'])
print('Atletas:', atletas.shape)

print('Atletasreg:', atletasreg.shape)
atletasreg.info()
atletasreg.sample(5).T
region_nulos = atletasreg[atletasreg['region'].isnull()]
region_nulos.sample(5).T
region_nulos = region_nulos[['NOC', 'Team', 'ID']].groupby(['NOC', 'Team']).sum().reset_index()
region_nulos.info()
region_nulos.drop('ID', axis=1, inplace=True)
region_nulos.head(8)
region_nulos.drop(0, inplace=True)
region_nulos.drop(1, inplace=True)
region_nulos.drop(3, inplace=True)
region_nulos.drop(4, inplace=True)
region_nulos.rename(columns={'Team':'region'}, inplace=True)
region_nulos.head(4)
regioes = regioes.append(region_nulos, ignore_index=True)
atletasreg = pd.merge(atletas[atletas["Season"] != 'Winter'], regioes, how='left', on=['NOC', 'NOC'])
atletasreg.info()
atletasreg.rename(columns={'Age':'Idade'}, inplace=True)

atletasreg.rename(columns={'City':'Cidade'}, inplace=True)

atletasreg.rename(columns={'Event':'Modalidade'}, inplace=True)

atletasreg.rename(columns={'Games':'Jogos'}, inplace=True)

atletasreg.rename(columns={'Height':'Altura'}, inplace=True)

atletasreg.rename(columns={'Medal':'Medalha'}, inplace=True)

atletasreg.rename(columns={'Name':'Nome'}, inplace=True)

atletasreg.rename(columns={'NOC':'SigladoPais'}, inplace=True)

atletasreg.rename(columns={'region':'Pais'}, inplace=True)

atletasreg.rename(columns={'Season':'Estacao'}, inplace=True)

atletasreg.rename(columns={'Sex':'Sexo'}, inplace=True)

atletasreg.rename(columns={'Sport':'Esporte'}, inplace=True)

atletasreg.rename(columns={'Team':'Equipe'}, inplace=True)

atletasreg.rename(columns={'Weight':'Peso'}, inplace=True)

atletasreg.rename(columns={'Year':'Ano'}, inplace=True)
atletasreg.head(5).T
atletasreg[atletasreg['Pais'] == 'Brazil']
atletasreg[atletasreg['Equipe'] == 'Demi-Mondaine-17']
def atribuiouro(s):

    if s == "Gold":

       return 1

    else:

       return 0
atletasreg['Ouro'] = atletasreg['Medalha'].apply(atribuiouro)
atletasreg['Ouro'].sum()
def atribuiprata(s):

    if s == "Silver":

       return 1

    else:

       return 0
atletasreg['Prata'] = atletasreg['Medalha'].apply(atribuiprata)
atletasreg['Prata'].sum()
def atribuibronze(s):

    if s == "Bronze":

       return 1

    else:

       return 0
atletasreg['Bronze'] = atletasreg['Medalha'].apply(atribuibronze)
atletasreg['Bronze'].sum()
atletasreg['Total'] = atletasreg['Ouro'] + atletasreg['Prata'] + atletasreg['Bronze']
atletasreg.sample(10).T
Paises = atletasreg[['Pais', 'SigladoPais', 'Equipe', 'Ouro', 'Prata', 'Bronze']].groupby(['Pais', 'SigladoPais', 'Equipe']).sum().reset_index()
#Paises.to_excel("PaisesA.xlsx", sheet_name='PaisesF')
paises_trad = pd.read_excel('/kaggle/input/tradcp/Paises-Trad.xlsx') #, sheet='PaisesA-Tradução')
paises_trad.info()
atletasreg = pd.merge(atletasreg, paises_trad, how='left', on=['Pais', 'Pais'])
atletasreg.info()
atletasreg = atletasreg.fillna({"Altura": atletasreg['Altura'].mean()//1, "Idade": atletasreg['Idade'].mean()//1,  "Peso": atletasreg['Peso'].mean()//1})
atletasreg.nlargest(50, 'Altura')
Esportes = atletasreg[['Esporte', 'ID']].groupby(['Esporte']).sum().reset_index()
#Esportes.to_excel("/kaggle/input/tradcp/Esportes.xlsx", sheet_name='Trad')
Esportes_trad = pd.read_excel('/kaggle/input/tradcp/Esportes-Trad.xlsx')
Esportes_trad.info()
atletasreg = pd.merge(atletasreg, Esportes_trad, how='left', on=['Esporte', 'Esporte'])
atletasreg.info()
atletasreg.drop('notes', axis=1, inplace=True)
atletasreg['Altura'] = atletasreg['Altura']/100
atletasreg['Estacao_PT_BR'] = 'Verão'
atletasreg.Estacao_PT_BR.value_counts()
Modalidades = atletasreg[['Modalidade', 'Ouro']].groupby(['Modalidade']).sum().reset_index()
#Modalidades.to_excel("Modalidades.xlsx", sheet_name='Trad')
Modalidades_trad = pd.read_excel('/kaggle/input/tradcp/Modalidades-Trad.xlsx')
Modalidades_trad.info()
atletasreg = pd.merge(atletasreg, Modalidades_trad, how='left', on=['Modalidade', 'Modalidade'])
atletasreg.tail(5).T
Cidades = atletasreg[['Cidade', 'Ano']].groupby(['Cidade']).mean().reset_index()
#Cidades.to_excel("Cidades.xlsx", sheet_name='Trad')
Cidades_trad = pd.read_excel('/kaggle/input/tradcp/Cidades-Trad.xlsx')
atletasreg = pd.merge(atletasreg, Cidades_trad, how='left', on=['Cidade', 'Cidade'])
atletasreg.head(5).T
%matplotlib inline
qmI = atletasreg
qmI.info()
qmI.drop('ID', axis=1, inplace=True)

qmI.drop('Nome', axis=1, inplace=True)

qmI.drop('Sexo', axis=1, inplace=True)

qmI.drop('Idade', axis=1, inplace=True)

qmI.drop('Altura', axis=1, inplace=True)

qmI.drop('Peso', axis=1, inplace=True)
qmI.drop_duplicates(inplace=True)
qmV = qmI[['Ano', 'Pais-PT-BR', 'Ouro', 'Prata', 'Bronze', 'Total']].groupby(['Ano', 'Pais-PT-BR']).sum().reset_index()
qmV[qmV['Pais-PT-BR'] == 'Estados Unidos']
qmT = qmV[qmV['Pais-PT-BR'] == 'Estados Unidos']
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Ouro', data=qmT, color='green')

plt.title('As medalhas de Ouro dos Estados Unidos nas Olímpiadas')

plt.locator_params(axis='y', nbins=30) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Prata', data=qmT, color='green')

plt.title('As medalhas de Prata dos Estados Unidos nas Olímpiadas')

plt.locator_params(axis='y', nbins=30) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Bronze', data=qmT, color='green')

plt.title('As medalhas de Bronze dos Estados Unidos nas Olímpiadas')

plt.locator_params(axis='y', nbins=20) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,10))

sns.pointplot(x='Ano', y='Total', data=qmT, color='green')

plt.title('O total de medalhas dos Estados Unidos nas Olímpiadas')

plt.locator_params(axis='y', nbins=20) #diminuindo a escala de y

plt.grid(True, color='grey')
qmY = qmI[['Ano', 'Ouro', 'Prata', 'Bronze', 'Total']].groupby(['Ano']).sum().reset_index()
qmY.head(50)
qmB = qmV[qmV['Pais-PT-BR'] == 'Brasil']
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Ouro', data=qmB, color='green')

plt.title('As medalhas de Ouro do Brasil nas Olímpiadas')

plt.locator_params(axis='y', nbins=10) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Prata', data=qmB, color='green')

plt.title('As medalhas de Prata do Brasil nas Olímpiadas')

plt.locator_params(axis='y', nbins=10) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Bronze', data=qmB, color='green')

plt.title('As medalhas de Bronze do Brasil nas Olímpiadas')

plt.locator_params(axis='y', nbins=12) #diminuindo a escala de y

plt.grid(True, color='grey')
plt.figure(figsize=(15,10))

sns.pointplot(x='Ano', y='Total', data=qmB, color='green')

plt.title('O total de medalhas do Brasil nas Olímpiadas')

plt.locator_params(axis='y', nbins=27) #diminuindo a escala de y

plt.grid(True, color='grey')
#qmI[['Ano', 'Pais-PT-BR', 'Ouro', 'Prata', 'Bronze', 'Total']].groupby(['Ano', 'Pais-PT-BR']).sum().reset_index()

qmOuroB = qmV[qmV['Pais-PT-BR'] == 'Brasil']

qmOuroE = qmV[qmV['Pais-PT-BR'] == 'Estados Unidos']

qmOuroC = qmV[qmV['Pais-PT-BR'] == 'Cuba']

qmOuroR = qmV[qmV['Pais-PT-BR'] == 'Rússia']

qmAno = qmY

qmOuroB.info()
qmOuroB.drop('Prata', axis=1, inplace=True)

qmOuroB.drop('Bronze', axis=1, inplace=True)

qmOuroB.drop('Total', axis=1, inplace=True)

qmOuroE.drop('Prata', axis=1, inplace=True)

qmOuroE.drop('Bronze', axis=1, inplace=True)

qmOuroE.drop('Total', axis=1, inplace=True)

qmOuroC.drop('Prata', axis=1, inplace=True)

qmOuroC.drop('Bronze', axis=1, inplace=True)

qmOuroC.drop('Total', axis=1, inplace=True)

qmOuroR.drop('Prata', axis=1, inplace=True)

qmOuroR.drop('Bronze', axis=1, inplace=True)

qmOuroR.drop('Total', axis=1, inplace=True)

qmAno.drop('Prata', axis=1, inplace=True)

qmAno.drop('Bronze', axis=1, inplace=True)

qmAno.drop('Total', axis=1, inplace=True)

qmAno.drop('Ouro', axis=1, inplace=True)
qmOuroB.rename(columns={'Ouro':'Brasil'}, inplace=True)

qmOuroE.rename(columns={'Ouro':'EUA'}, inplace=True)

qmOuroC.rename(columns={'Ouro':'Cuba'}, inplace=True)

qmOuroR.rename(columns={'Ouro':'Rússia'}, inplace=True)

qmOuroB.drop('Pais-PT-BR', axis=1, inplace=True)

qmOuroE.drop('Pais-PT-BR', axis=1, inplace=True)

qmOuroC.drop('Pais-PT-BR', axis=1, inplace=True)

qmOuroR.drop('Pais-PT-BR', axis=1, inplace=True)
qmOuro = pd.merge(qmAno, qmOuroB, how='left', on=['Ano', 'Ano'])

qmOuro = pd.merge(qmOuro, qmOuroE, how='left', on=['Ano', 'Ano'])

qmOuro = pd.merge(qmOuro, qmOuroC, how='left', on=['Ano', 'Ano'])

qmOuro = pd.merge(qmOuro, qmOuroR, how='left', on=['Ano', 'Ano'])
qmOuro.head(50)
qmOuro = qmOuro.fillna({"Brasil": qmOuro['Brasil'].min(), "EUA": 0,  "Cuba": qmOuro['Cuba'].min(), "Rússia": qmOuro['Rússia'].min()})

qmOuro.head(50)
ax = plt.gca()

qmOuro.plot(kind='line',x='Ano',y='Brasil',color='green', ax=ax)

qmOuro.plot(kind='line',x='Ano',y='EUA', color='blue', ax=ax)

qmOuro.plot(kind='line',x='Ano',y='Rússia', color='red', ax=ax)

qmOuro.plot(kind='line',x='Ano',y='Cuba', color='Yellow', ax=ax)

plt.show()