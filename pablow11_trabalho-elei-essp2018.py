# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as py

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importação das bases de dados

candidato = pd.read_csv('../input/candidato.csv', sep=';', encoding='latin1')

despesas = pd.read_csv('../input/despesas.csv', sep=';', encoding='latin1')

receitas = pd.read_csv('../input/receitas.csv', sep=';', encoding='latin1')

votos = pd.read_csv('../input/votos.csv', sep=';', encoding='latin1')
# Dimensões das bases de dados

print('Dimensões - Candidatos: ', candidato.shape)

print('Dimensões - Despesas: ', despesas.shape)

print('Dimensões - Receitas: ', receitas.shape)

print('Dimensões - Votos: ', votos.shape)
# Base candidato

candidato.head()
# Base despesas

despesas.head()
# Base receitas

receitas.head()
# Base Votos

votos.head()
# Base receitas



# Exclusão da receita originario conforme atualização efetuada pelo professor em 02/04/2019

receitas.drop(columns=['SQ_RECEITA', 'VR_RECEITA_ORIGINARIO'],inplace=True)



# Conversão do campo VR_RECEITA para ponto flutuante

receitas['VR_RECEITA'] = receitas['VR_RECEITA'].str.replace('.','').str.replace(',','.').astype(float)

# Agrupar

receitas = receitas.groupby('SQ_CANDIDATO').sum()



# Exibir

receitas.head()
# Base despesas



# Conversão dos campos Despesa contratada e despesa paga para ponto flutuante

despesas['VR_DESPESA_CONTRATADA'] = despesas['VR_DESPESA_CONTRATADA'].str.replace('.','').str.replace(',','.').astype(float)

despesas['VR_PAGTO_DESPESA'] = despesas['VR_PAGTO_DESPESA'].str.replace('.','').str.replace(',','.').astype(float)



despesas.info()
despesas.head()
# Base votos



# Conversão do campo votos nominais paga para ponto flutuante

votos['QT_VOTOS_NOMINAIS'] = votos['QT_VOTOS_NOMINAIS'].str.replace('.','').str.replace(',','.').astype(float)

votos.info()
votos.head()
# Aplicação de filtros



# Manutenção dos candidatos com situação APTA

candidato = candidato[candidato.CD_SITUACAO_CANDIDATURA == 12]



# Outros filtros podem ser adicionados aqui

# Junção das bases 

cand = pd.merge(candidato, despesas, on='SQ_CANDIDATO', how='left')

cand = pd.merge(cand, receitas, on='SQ_CANDIDATO', how='left')

cand = pd.merge(cand, votos, on='SQ_CANDIDATO', how='left')
# A base a ser utilizada é a cand!!!!!



cand.info()
cand.head()
# Correlação entre as variáveis numéricas

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(cand.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=4)

plt.show()
# Observa-se correlação positiva entre

# Votos e receita (0.81)

# Votos e despesas contratadas (0.78)

# Votos e despesas pagas (0.74)

# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['VR_DESPESA_CONTRATADA'], y=cand['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Votos em relação aos gastos de campanha - DEP.FEDERAL",color='black')

plt.xlabel('Despesas')

plt.ylabel("Quantidade de votos")

plt.show()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['VR_DESPESA_CONTRATADA'], y=cand['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Votos em relação aos gastos de campanha - DEP.ESTADUAL",color='black')

plt.xlabel('Despesas')

plt.ylabel("Quantidade de votos")

plt.show()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand[cand['DS_CARGO'] == 'GOVERNADOR']['VR_DESPESA_CONTRATADA'], y=cand['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Votos em relação aos gastos de campanha - GOVERNADOR",color='black')

plt.xlabel('Despesas')

plt.ylabel("Quantidade de votos")

plt.show()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand[cand['DS_CARGO'] == 'SENADOR']['VR_DESPESA_CONTRATADA'], y=cand['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Votos em relação aos gastos de campanha - SENADOR",color='black')

plt.xlabel('Despesas')

plt.ylabel("Quantidade de votos")

plt.show()
# Despesas por cargo

plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['VR_DESPESA_CONTRATADA'].dropna(), color='blue', label='DEPUTADO ESTADUAL')

plt.title("Distribuição das despesas por cargo",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.show()
plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['VR_DESPESA_CONTRATADA'].dropna(), color='green', label='DEPUTADO FEDERAL')

plt.title("Distribuição das despesas por cargo",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.show()
plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'GOVERNADOR']['VR_DESPESA_CONTRATADA'].dropna(), color='purple', label='GOVERNADOR')

plt.title("Distribuição das despesas por cargo",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.show()
plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'SENADOR']['VR_DESPESA_CONTRATADA'].dropna(), color='black', label='SENADOR')

plt.title("Distribuição das despesas por cargo",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.show()
# Distribuição dos candidatos por despesas por cargo (boxplot)

plt.figure(figsize=(12,8))

sns.boxplot(x='DS_CARGO', y='VR_DESPESA_CONTRATADA', data=cand)

plt.title("Despesas por cargo",color='black')

plt.ylabel("Despesas")

plt.xlabel("Cargo")

plt.xticks(rotation='vertical')

plt.show()
# Distribuição dos candidatos por faixa etária (boxplot)

plt.figure(figsize=(12,8))

sns.boxplot(x='DS_CARGO', y='NR_IDADE_DATA_POSSE', data=cand)

plt.title("Faixa etária por cargo",color='black')

plt.ylabel("Faixa etária")

plt.xlabel("Cargo")

plt.xticks(rotation='vertical')

plt.show()
cand.info()
cand['DS_CARGO'].unique()
# Distribuição dos candidatos por faixa etária

plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['NR_IDADE_DATA_POSSE'].dropna(), color='blue', label='DEPUTADO ESTADUAL')

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['NR_IDADE_DATA_POSSE'].dropna(), color='green', label='DEPUTADO FEDERAL')

sns.distplot(cand[cand['DS_CARGO'] == '2º SUPLENTE']['NR_IDADE_DATA_POSSE'].dropna(), color='red', label='2º SUPLENTE')

sns.distplot(cand[cand['DS_CARGO'] == '1º SUPLENTE']['NR_IDADE_DATA_POSSE'].dropna(), color='yellow', label='1º SUPLENTE')

sns.distplot(cand[cand['DS_CARGO'] == 'VICE-GOVERNADOR']['NR_IDADE_DATA_POSSE'].dropna(), color='orange', label='VICE-GOVERNADOR')

sns.distplot(cand[cand['DS_CARGO'] == 'GOVERNADOR']['NR_IDADE_DATA_POSSE'].dropna(), color='purple', label='GOVERNADOR')

sns.distplot(cand[cand['DS_CARGO'] == 'SENADOR']['NR_IDADE_DATA_POSSE'].dropna(), color='black', label='SENADOR')

plt.title("Distribuição dos candidatos por faixa etária",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.xlabel('Idade na data da posse')

plt.ylabel('Quantidade de candidatos')

plt.show()
#Pode-se perceber que há semelhança entre os grupos analisados com concentração dos candidatos na faixa etária entre 40 e 60 anos.

# Percebe-se também que para 1º suplente e vice-governador há uma maior concentração entre 40 e 50 anos

# e para senador, governado e 2º suplente ha uma maior concentração entre os 55 e 65
# Distribuição dos candidatos por gênero

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_GENERO', data=cand)

plt.xticks(rotation=90)

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de candidatos por gênero",color='black')

plt.legend(title="Sexo")

plt.xlabel('Cargo')

plt.ylabel("Quantidade de candidatos")

plt.show()
# Total de votos em relação ao gênero do candidato

plt.figure(figsize=(5,10))

sns.boxplot(x='DS_GENERO', y='QT_VOTOS_NOMINAIS', data=cand)

plt.title("Votação por gênero",color='black')

plt.xlabel('Gênero')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos candidatos por formação acadêmica

plt.figure(figsize=(15,5))

sns.countplot(x='DS_CARGO', hue='DS_GRAU_INSTRUCAO', data=cand)

plt.title("Distribuição dos candidatos por formação acadêmica",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Grau de Instrução")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
# Total de votos em relação ao grau de instrução

plt.figure(figsize=(5,10))

sns.boxplot(x='DS_GRAU_INSTRUCAO', y='QT_VOTOS_NOMINAIS', data=cand)

plt.xticks(rotation=90)

plt.title("Votação por formação acadêmica",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Grau de instrução')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos candidatos por raça

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_COR_RACA', data=cand)

plt.title("Quantidade de candidatos por raça",color='black')

plt.xticks(rotation=90)

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Raça")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
# Distribuição dos candidatos por estado civil

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_ESTADO_CIVIL', data=cand)

plt.title("Quantidade de candidatos estado civil",color='black')

plt.xticks(rotation=90)

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Estado Civil")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
# Distribuição dos candidatos por estado civil

plt.figure(figsize=(10,5))

sns.countplot(x='DS_GENERO', hue='DS_NACIONALIDADE', data=cand)

plt.title("Quantidade de candidatos estado civil",color='black')

plt.xticks(rotation=90)

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Estado Civil")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
#Distribuição do Valor em despesa por cargo por situação do turno

plt.figure(figsize=(15,5))

sns.barplot(x='DS_CARGO', y='VR_DESPESA_CONTRATADA', hue='DS_SIT_TOT_TURNO', data=cand, ci=None)

plt.xlabel('Cargo')

plt.ylabel("Valor da despesa (R$)")

plt.title("Valor em despesa por cargo por situação do turno")

plt.legend(title="Situação do turno")

plt.axhline(y=cand['VR_DESPESA_CONTRATADA'].mean())
#Distribuição do Valor em despesa por cargo por situação do turno

plt.figure(figsize=(15,5))

sns.barplot(x='DS_SIT_TOT_TURNO', y='VR_DESPESA_CONTRATADA', hue='DS_GENERO', data=cand, ci=None)

plt.xlabel('Cargo')

plt.ylabel("Valor da despesa (R$)")

plt.title("Valor em despesa por cargo por situação do turno")

plt.legend(title="Situação do turno")

plt.axhline(y=cand['VR_DESPESA_CONTRATADA'].mean())
#Distribuição do Valor em despesa por cargo por situação do turno

plt.figure(figsize=(15,5))

sns.barplot(x='DS_SIT_TOT_TURNO', y='QT_VOTOS_NOMINAIS', hue='DS_GENERO', data=cand, ci=None)

plt.xlabel('Cargo')

plt.ylabel("Valor da despesa (R$)")

plt.title("Valor em despesa por cargo por situação do turno")

plt.legend(title="Situação do turno")

plt.axhline(y=cand['QT_VOTOS_NOMINAIS'].mean())
cand['DS_NACIONALIDADE'].unique()
cand.columns
# Maiores despesas por partido



sns.countplot(x='SG_PARTIDO', data=cand, order=cand['Winner'].value_counts().index)

plt.title("Despesas por partido",color='black')

plt.show()



# Total de gols por seleção

desp_sg_partido = cand.groupby("SG_PARTIDO")["VR_DESPESA_CONTRATADA"].sum().reset_index()

gols_casa.columns = ["selecao","gols"]

gols["gols"] = gols["gols"].astype(int)



plt.figure(figsize=(15,5))

sns.barplot(x="selecao",y="gols", data=gols[:10])

plt.title("Seleções com maior número de gols marcados",color='black')

plt.show()
desp_sg_partido = cand.groupby("SG_PARTIDO")["VR_DESPESA_CONTRATADA"].sum().reset_index()
desp_sg_partido.info()
cand.to_csv('cand_final.csv', sep=';', encoding='latin1', index=False)
# GERA LINK PARA BAIXAR O CSV SEM PRECISAR COMPILAR

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "cand_final.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='cand_final.csv')