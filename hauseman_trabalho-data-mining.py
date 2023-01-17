# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mg_candidatos = pd.read_excel('../input/Consulta_Candidatos.xlsx', encoding='latin1', sep=';')
mg_votos = pd.read_excel('../input/votacao_candidato.xlsx', encoding='latin1', sep=';')
mg_desp_contratadas = pd.read_excel('../input/despesas_contratadas.xlsx', encoding='latin1', sep=';')
mg_receitas = pd.read_excel('../input/receitas_candidatos.xlsx', encoding='latin1', sep=';')
mg_desp_pagas = pd.read_excel('../input/despesas_pagas.xlsx', encoding='latin1', sep=';')
mg_candidatos.info()
mg_desp_pagas.info()
mg_receitas.info()
mg_votos.info()
mg_candidatos.info()
mg_candidatos.head()
# Na base de dados, candidatos que vão para o segundo turno são registrados duas vezes. É necessário excluir este registro afim 

# de se evitar duplicidades, inconsistências (CD_SIT_TOT_TURNO = 6)

mg_candidatos = mg_candidatos[mg_candidatos.CD_SIT_TOT_TURNO != 6]



# Serão mantidos apenas os candidatos com candidatura APTA CD_SITUACAO_CANDIDATURA == 12

mg_candidatos = mg_candidatos[mg_candidatos.CD_SITUACAO_CANDIDATURA == 12]



# Setando apenas deputados federais / estaduais

# ESTADUAL CD_CARGO = 8

# FEDERAL CD_CARGO = 6

# SENADOR = 5

mg_candidatos = mg_candidatos[(mg_candidatos.CD_CARGO == 5) | (mg_candidatos.CD_CARGO == 6) | (mg_candidatos.CD_CARGO == 8)]
mg_desp_pagas.head()
mg_desp_contratadas.head()
# Formatação dos dados do campo VR_BEM_CANDIDATO. Os dados em formato numérico nacional podem apresentar conflitos com o 

# padrão americano.

#mg_desp_pagas['VR_PAGTO_DESPESA'] = mg_desp_pagas['VR_PAGTO_DESPESA'].astype(str).str.replace(',','.').astype(float)



#dc_listings['price'].astype(str).str.replace



mg_desp_contratadas['VR_DESPESA_CONTRATADA'] = mg_desp_contratadas['VR_DESPESA_CONTRATADA'].astype(str).str.replace(',','.').astype(float)
mg_desp_contratadas.head()
# Somatório das despesas

mg_desp_contratadas_GROUP = mg_desp_contratadas[['NR_CANDIDATO', 'VR_DESPESA_CONTRATADA']]

mg_desp_contratadas_GROUP = mg_desp_contratadas.groupby('NR_CANDIDATO').sum()
mg_desp_contratadas_GROUP.head()
# União das tabelas

mg_candidatos = pd.merge(mg_candidatos, mg_desp_contratadas_GROUP, on='NR_CANDIDATO', how='left')

mg_candidatos = pd.merge(mg_candidatos, mg_votos, on='NR_CANDIDATO', how='left')
mg_candidatos.head()
mask = (mg_candidatos['DS_ESTADO_CIVIL'] == "SEPARADO(A) JUDICIALMENTE") | (mg_candidatos['DS_ESTADO_CIVIL'] == "DIVORCIADO(A)")

mg_candidatos['DS_ESTADO_CIVIL'] = mg_candidatos['DS_ESTADO_CIVIL'].mask(mask, "SEPARADO")



mask = (mg_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL INCOMPLETO") | (mg_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL COMPLETO")| (mg_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO INCOMPLETO")

mg_candidatos['DS_GRAU_INSTRUCAO'] = mg_candidatos['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO FUNDAMENTAL")



mask = (mg_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO COMPLETO") | (mg_candidatos['DS_GRAU_INSTRUCAO'] == "SUPERIOR INCOMPLETO")

mg_candidatos['DS_GRAU_INSTRUCAO'] = mg_candidatos['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO MÉDIO")
mg_candidatos.head()
# Distribuição dos candidatos por faixa etária

plt.figure(figsize=(15,5))

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['NR_IDADE_DATA_POSSE'].dropna(), color='blue', label='DEPUTADO ESTADUAL')

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['NR_IDADE_DATA_POSSE'].dropna(), color='green', label='DEPUTADO FEDERAL')

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO DISTRITAL']['NR_IDADE_DATA_POSSE'].dropna(), color='red', label='DEPUTADO DISTRITAL')

sns.distplot(mg_candidatos['NR_IDADE_DATA_POSSE'])

plt.title("Distribuição dos candidatos por faixa etária",color='black')

plt.grid(True,color="grey",alpha=.3)

#plt.legend(title="Cargo")

plt.xlabel('Idade na data da posse')

plt.ylabel('Distribuição')

plt.show()
# Distribuição dos candidatos por gênero

plt.figure(figsize=(15,8))

sns.countplot(x='CD_CARGO', hue='DS_GENERO', data=mg_candidatos)

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de candidatos por gênero",color='black')

plt.legend(title="Sexo")

plt.xlabel('Cargo')

plt.ylabel("Quantidade de candidatos")

plt.show()
mg_candidatos.info()
# Distribuição dos candidatos por raça

plt.figure(figsize=(10,5))

sns.countplot(x='DS_COR_RACA', data=mg_candidatos)

plt.title("Quantidade de candidatos por raça",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Raça')

plt.ylabel("Quantidade")

plt.show()
# Criação da variável TARGET para análise de dados. CD_SIT_TOT_TURNO = 1 ELEITO, 2 ELEITO POR QP e 3 ELEITO POR MÉDIA



ELEITO = []



for i in mg_candidatos['CD_SIT_TOT_TURNO']:

    if i == 1:

        ELEITO.append('sim')

    elif i == 2:

        ELEITO.append('sim')

    elif i == 3:

        ELEITO.append('sim')

    else:

        ELEITO.append('nao')

        

mg_candidatos['ELEITO'] = ELEITO



#mg_eleito = mg_candidatos[(mg_candidatos.DS_SIT_TOT_TURNO == 'ELEITO POR MÉDIA')

#                         |(mg_candidatos.DS_SIT_TOT_TURNO == 'ELEITO')

#                         |(mg_candidatos.DS_SIT_TOT_TURNO == 'ELEITO POR QP')]
mg_eleitos = mg_candidatos[mg_candidatos['ELEITO'] == 'sim']

mg_eleitos.info()
plt.figure(figsize=(15,18))

sns.countplot(y='NM_PARTIDO', data=mg_eleitos, order = mg_eleitos['NM_PARTIDO'].value_counts().index)

plt.title("Eleitos por partido",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Quantidade de candidatos eleitos')

plt.ylabel("Partido")

plt.show()
mg_candidatos.info()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=mg_candidatos['VR_DESPESA_CONTRATADA']/10, y=mg_candidatos['q'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos votos aos gastos em campanha",color='black')

plt.xlabel('Valor das despesas em Reais (R$)')

plt.ylabel("Quantidade de votos")

plt.show()
# Custo por voto dos candidatos eleitos



eleitos = mg_candidatos[mg_candidatos['ELEITO'] == 'sim']

eleitos = eleitos[['VR_DESPESA_CONTRATADA', 'QT_VOTOS_NOMINAIS', 'DS_CARGO']]



eleitos['CUSTO_VOTO'] = eleitos['VR_DESPESA_CONTRATADA']/eleitos['QT_VOTOS_NOMINAIS']



eleitos = eleitos.groupby(['DS_CARGO']).sum().reset_index()
eleitos.head()
plt.figure(figsize=(15,5))

sns.barplot(x='DS_CARGO', y='CUSTO_VOTO', data=eleitos, ci=None)

plt.xlabel('Cargo')

plt.ylabel("Custo por voto em Reias (R$)")

plt.title("Custo por voto para os eleitos")

plt.axhline(y=eleitos['CUSTO_VOTO'].mean())
# Número de votos em relação à despesa Deputado Estadual

mg_deputados_estaduais = mg_candidatos[mg_candidatos['CD_CARGO'] == 8]

plt.figure(figsize=(15,5))

sns.scatterplot(x=mg_deputados_estaduais['VR_DESPESA_CONTRATADA']/10, y=mg_deputados_estaduais['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True, color="grey",alpha=.3)

plt.title("Distribuição dos votos aos gastos em campanha",color='black')

plt.xlabel('Valor das despesas em Reais (R$)')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos candidatos por bens declarados (boxplot)

mg_deputados = mg_candidatos[(mg_candidatos.CD_CARGO == 6) | (mg_candidatos.CD_CARGO == 8)]

plt.figure(figsize=(15,10))

sns.boxplot(x='DS_CARGO', y='VR_DESPESA_CONTRATADA', data=mg_deputados)

plt.title("Candidatos por faixa etária",color='black')

plt.ylabel("Idade do candidato")

plt.xlabel("Cargo")

plt.show()
mg_candidatos.sort_values(by=['VR_DESPESA_CONTRATADA'], ascending=False)

#eleitos['QT_VOTOS_NOMINAIS'] = eleitos['QT_VOTOS_NOMINAIS'].astype(int)