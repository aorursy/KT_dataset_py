# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as time

import datetime as datetime

import deltatime as deltatime







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importando os dados para DataFrames

dfLondomMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_London.csv", delimiter=',', encoding ='unicode_escape')

dfLondomFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_London.csv", delimiter=',', encoding ='unicode_escape')



dfBostonMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_Boston.csv")

dfBostonFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_Boston.csv")



dfNYMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_New_York.csv", delimiter=',', encoding ='unicode_escape')

dfNYFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_New_York.csv")



dfChicagoMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_Chicago.csv", delimiter=',', encoding ='unicode_escape')

dfChicagoFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_Chicago.csv")



dfBerlinMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_Berlin.csv", delimiter=',', encoding ='unicode_escape')

dfBerlinFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_Berlin.csv", delimiter=',', encoding ='unicode_escape')



dfTokioMale = pd.read_csv("/kaggle/input/world-marathon-majors/Male_Elite_Tokyo.csv", delimiter=',', encoding ='unicode_escape')

dfTokioFemale = pd.read_csv("/kaggle/input/world-marathon-majors/Female_Elite_Tokyo.csv")
# Alteração dos nomes dos campos - para português

dfLondomMale.rename(columns={'Year':'Ano'}, inplace=True)

dfLondomMale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfLondomMale.rename(columns={'Nationality':'Nacionalidade'}, inplace=True)

dfLondomMale.rename(columns={'Time(h:m:s)':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfLondomMale['Sexo'] = "M"

dfLondomMale['Prova'] = "Londres"



# Deletando o campo State, não será utilizado nesse estudo

del dfLondomMale['Notes']



dfLondomMale.info()
dfLondomMale.shape
# Alteração dos nomes dos campos - para português

dfLondomFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfLondomFemale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfLondomFemale.rename(columns={'Nationality':'Nacionalidade'}, inplace=True)

dfLondomFemale.rename(columns={'Time(h:m:s)':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfLondomFemale['Sexo'] = "F"

dfLondomFemale['Prova'] = "Londres"



# Deletando o campo State, não será utilizado nesse estudo

del dfLondomFemale['Notes']





dfLondomFemale.info()
dfLondomFemale.shape
# Alteração dos nomes dos campos - para português

dfBostonMale.rename(columns={'Year':'Ano'}, inplace=True)

dfBostonMale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfBostonMale.rename(columns={'Country/State or Province':'Nacionalidade'}, inplace=True)

dfBostonMale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfBostonMale['Sexo'] = "M"

dfBostonMale['Prova'] = "Boston"



# Deletando os campos State e Notes, não será utilizado nesse estudo

del dfBostonMale['State']

del dfBostonMale['Notes']



dfBostonMale.info()
dfBostonMale.shape
dfBostonFemale.info()
# Alteração dos nomes dos campos - para português

dfBostonFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfBostonFemale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfBostonFemale.rename(columns={'Country/State':'Nacionalidade'}, inplace=True)

dfBostonFemale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfBostonFemale['Sexo'] = "F"

dfBostonFemale['Prova'] = "Boston"



# Deletando os campos State e Notes, não será utilizado nesse estudo

del dfBostonFemale['State']

del dfBostonFemale['Notes']



dfBostonFemale.info()
dfBostonFemale.shape
# Alteração dos nomes dos campos - para português

dfNYMale.rename(columns={'Year':'Ano'}, inplace=True)

dfNYMale.rename(columns={'Winner':'NomeAtleta'}, inplace=True)

dfNYMale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfNYMale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfNYMale['Sexo'] = "M"

dfNYMale['Prova'] = "Nova Iorque"



# Deletando o campo Notes, não será utilizado nesse estudo

del dfNYMale['Notes']



dfNYMale.info()
dfNYMale.shape
# Alteração dos nomes dos campos - para português

dfNYFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfNYFemale.rename(columns={'Winner':'NomeAtleta'}, inplace=True)

dfNYFemale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfNYFemale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfNYFemale['Sexo'] = "F"

dfNYFemale['Prova'] = "Nova Iorque"



# Deletando o campo Notes, não será utilizado nesse estudo

del dfNYFemale['Notes']



dfNYFemale.info()
dfNYFemale.shape
# Alteração dos nomes dos campos - para português

dfChicagoMale.rename(columns={'Year':'Ano'}, inplace=True)

dfChicagoMale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfChicagoMale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfChicagoMale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfChicagoMale['Sexo'] = "M"

dfChicagoMale['Prova'] = "Chicago"



dfChicagoMale.info()
dfChicagoMale.shape
# Alteração dos nomes dos campos - para português

dfChicagoFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfChicagoFemale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfChicagoFemale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfChicagoFemale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfChicagoFemale['Sexo'] = "F"

dfChicagoFemale['Prova'] = "Chicago"



dfChicagoFemale.info()
dfChicagoFemale.shape
dfBerlinMale.rename(columns={'Year':'Ano'}, inplace=True)

dfBerlinMale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfBerlinMale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfBerlinMale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfBerlinMale['Sexo'] = "M"

dfBerlinMale['Prova'] = "Berlin"



dfBerlinMale.info()
dfBerlinMale.shape
# Alteração dos nomes dos campos - para português

dfBerlinFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfBerlinFemale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfBerlinFemale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfBerlinFemale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfBerlinFemale['Sexo'] = "F"

dfBerlinFemale['Prova'] = "Berlin"



dfBerlinFemale.info()
dfBerlinFemale.shape
# Alteração dos nomes dos campos - para português

dfTokioMale.rename(columns={'Year':'Ano'}, inplace=True)

dfTokioMale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfTokioMale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfTokioMale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfTokioMale['Sexo'] = "M"

dfTokioMale['Prova'] = "Tóquio"



dfTokioMale.info()
dfTokioMale.shape
# Alteração dos nomes dos campos - para português

dfTokioFemale.rename(columns={'Year':'Ano'}, inplace=True)

dfTokioFemale.rename(columns={'Athlete':'NomeAtleta'}, inplace=True)

dfTokioFemale.rename(columns={'Country':'Nacionalidade'}, inplace=True)

dfTokioFemale.rename(columns={'Time':'Tempo-hhmmss'}, inplace=True)



# Criar os campo "Sexo" (M - Masculino e F - Feminino) e "Prova" (Nome da prova)

dfTokioFemale['Sexo'] = "F"

dfTokioFemale['Prova'] = "Tóquio"



dfTokioFemale.info()
dfTokioFemale.shape
df = dfLondomMale

df = df.append(dfLondomFemale, ignore_index=True)

df = df.append(dfBostonMale, ignore_index=True)

df = df.append(dfBostonFemale, ignore_index=True)

df = df.append(dfNYMale, ignore_index=True)

df = df.append(dfNYFemale, ignore_index=True)

df = df.append(dfChicagoMale, ignore_index=True)

df = df.append(dfChicagoFemale, ignore_index=True)

df = df.append(dfBerlinMale, ignore_index=True)

df = df.append(dfBerlinFemale, ignore_index=True)

df = df.append(dfTokioMale, ignore_index=True)

df = df.append(dfTokioFemale, ignore_index=True)

df.info()

df.shape
#Criando a variavel tempo do tipo

df['Tempo'] = pd.to_timedelta(df['Tempo-hhmmss'])

df.info()
# Ordenando os Dados por ano, prova e sexo

df = df.sort_values(['Ano', 'Prova', 'Sexo'])

df.head(600)
# verificando o período que está sendo estudado 

df = df.sort_values(['Ano', 'Prova', 'Sexo'])

menorAno = df['Ano'].values.min()

maiorAno = df['Ano'].values.max()

print('Período analisado :', menorAno, maiorAno)
df['Sexo'].value_counts()
# Verificando a prova mais Antiga

MenorAno = df['Ano'].values.min()

dfProvaAntiga = df[df['Ano'] == MenorAno]

Prova = dfProvaAntiga['Prova'].values

print("Prova mais antiga ocorreu em : " , Prova , " no ano de " , str(MenorAno))
# Segregando Masculino

dfMas = df[df['Sexo'] == 'M']

dfMas.info()
# Segregando Feminino

dfFem = df[df['Sexo'] == 'F']

dfFem.info()
# Verificando em que ano as mulheres começaram a participar dessas provas

MenorAno = dfFem['Ano'].values.min()

print(MenorAno)
# Dados completos da prova

dfFem[dfFem['Ano']==dfFem['Ano'].values.min()]
# Melhor tempo das mulheres

dfFem[dfFem['Tempo-hhmmss'] == dfFem['Tempo-hhmmss'].values.min()]
# Atleta masculino que mais ganhou provas

dfMas[['NomeAtleta', 'Prova']].groupby(['NomeAtleta']).count().nlargest(1, 'Prova')
#Quais provas ele ganhou

dfMas[dfMas['NomeAtleta'] == 'Bill Rodgers']

# País que mais ganhou provas

dfMas[['Nacionalidade', 'Prova']].groupby(['Nacionalidade']).count().nlargest(1, 'Prova')
# 5 países 

dfMas[['Nacionalidade', 'Prova']].groupby(['Nacionalidade']).count().nlargest(5, 'Prova')
# Melhor tempo dos homens

dfRecordista = dfMas[dfMas['Tempo-hhmmss'] == dfMas['Tempo-hhmmss'].values.min()]

print(dfRecordista)
#Quais provas o recodista ganhou

nome = dfRecordista['NomeAtleta'].values

nome = nome[0]

dfRecordista = dfMas[dfMas['NomeAtleta'] == nome]

dfRecordista
# Melhor tempo dos homens

dfNaoRecordista = dfMas[dfMas['Tempo-hhmmss'] == dfMas['Tempo-hhmmss'].values.max()]

print(dfNaoRecordista)
%matplotlib inline
# Observando os tempos do recordista em ns

sns.barplot(y='Tempo' ,x='Prova', data=dfRecordista)
# Observando os tempos do recordista em ns

sns.barplot(y='Tempo' ,x='Prova', data=dfMas)
# box-plot das provas que Recordista ganhou

sns.violinplot(y='Ano', x='Prova', data=dfRecordista)
# Gráfico apresenta os anos de participação do recordista

plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='Prova', data=dfRecordista, color='green')

plt.title('Os anos que recordista foi campeão e prova')

plt.locator_params(axis='y', nbins=30) #diminuindo a escala de y

plt.grid(True, color='grey')