import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
df_candidatos = pd.read_csv('../input/consulta_cand_2018_DF.csv', encoding='latin1', sep=';')

df_votos = pd.read_csv('../input/votacao_candidato_munzona_2018_DF.csv', encoding='latin1', sep=';')

df_despesas_contratadas_candidatos = pd.read_csv('../input/despesas_contratadas_candidatos_2018_DF.csv', encoding='latin1', sep=';')

df_despesas_pagas_candidatos = pd.read_csv('../input/despesas_pagas_candidatos_2018_DF.csv', encoding='latin1', sep=';')                       
df_candidatos.info()
df_votos.info()
df_despesas_contratadas_candidatos.info()
df_despesas_pagas_candidatos.info()
# Para a análise serão mantidas apenas as colunas consideradas úteis ao estudo. 



df_candidatos.drop(columns=['DT_GERACAO', 'HH_GERACAO', 'CD_TIPO_ELEICAO', 'NM_TIPO_ELEICAO', 'NR_TURNO', 'CD_ELEICAO', 

                        'DS_ELEICAO', 'DT_ELEICAO', 'SG_UE', 'NR_CANDIDATO', 'NM_SOCIAL_CANDIDATO',

                        'NM_EMAIL', 'SQ_COLIGACAO', 

                        'DS_COMPOSICAO_COLIGACAO', 'SG_UF_NASCIMENTO', 'CD_MUNICIPIO_NASCIMENTO', 'NM_MUNICIPIO_NASCIMENTO', 

                        'DT_NASCIMENTO', 'NR_TITULO_ELEITORAL_CANDIDATO', 'NR_DESPESA_MAX_CAMPANHA', 'ST_DECLARAR_BENS',

                        'NR_PROTOCOLO_CANDIDATURA', 'NR_PROCESSO', 'ANO_ELEICAO', 'NR_CPF_CANDIDATO'],inplace=True) 
# Na base de dados, candidatos que vão para o segundo turno são registrados duas vezes. É necessário excluir este registro afim 

# de se evitar duplicidades, inconsistências (CD_SIT_TOT_TURNO = 6)

df_candidatos = df_candidatos[df_candidatos.CD_SIT_TOT_TURNO != 6]



# Serão mantidos apenas os candidatos com candidatura APTA CD_SITUACAO_CANDIDATURA == 12

df_candidatos = df_candidatos[df_candidatos.CD_SITUACAO_CANDIDATURA == 12]



# Setando apenas deputados federais / estaduais

# DISTRITAL CD_CARGO = 8

# FEDERAL CD_CARGO = 6

# SENADOR = 5

df_candidatos = df_candidatos[(df_candidatos.CD_CARGO == 5) | (df_candidatos.CD_CARGO == 6) | (df_candidatos.CD_CARGO == 8)]
df_despesas_pagas_candidatos.head()
df_despesas_contratadas_candidatos.head()
# Formatação dos dados do campo VR_BEM_CANDIDATO. Os dados em formato numérico nacional podem apresentar conflitos com o 

# padrão americano.

df_despesas_pagas_candidatos['VR_PAGTO_DESPESA'] = df_despesas_pagas_candidatos['VR_PAGTO_DESPESA'].str.replace(',','.').astype(float)

df_despesas_contratadas_candidatos['VR_DESPESA_CONTRATADA'] = df_despesas_contratadas_candidatos['VR_DESPESA_CONTRATADA'].str.replace(',','.').astype(float)
# Somatório das despesas

df_despesas_contratadas_candidatos_GROUP = df_despesas_contratadas_candidatos[['SQ_CANDIDATO', 'VR_DESPESA_CONTRATADA']]

df_despesas_contratadas_candidatos_GROUP = df_despesas_contratadas_candidatos_GROUP.groupby('SQ_CANDIDATO').sum()
# União das tabelas

df_candidatos = pd.merge(df_candidatos, df_despesas_contratadas_candidatos_GROUP, on='SQ_CANDIDATO', how='left')

df_candidatos = pd.merge(df_candidatos, df_votos, on='SQ_CANDIDATO', how='left')
mask = (df_candidatos['DS_ESTADO_CIVIL'] == "SEPARADO(A) JUDICIALMENTE") | (df_candidatos['DS_ESTADO_CIVIL'] == "DIVORCIADO(A)")

df_candidatos['DS_ESTADO_CIVIL'] = df_candidatos['DS_ESTADO_CIVIL'].mask(mask, "SEPARADO")



mask = (df_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL INCOMPLETO") | (df_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL COMPLETO")| (df_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO INCOMPLETO")

df_candidatos['DS_GRAU_INSTRUCAO'] = df_candidatos['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO FUNDAMENTAL")



mask = (df_candidatos['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO COMPLETO") | (df_candidatos['DS_GRAU_INSTRUCAO'] == "SUPERIOR INCOMPLETO")

df_candidatos['DS_GRAU_INSTRUCAO'] = df_candidatos['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO MÉDIO")
df_candidatos.head()
# Distribuição dos candidatos por faixa etária

plt.figure(figsize=(15,5))

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['NR_IDADE_DATA_POSSE'].dropna(), color='blue', label='DEPUTADO ESTADUAL')

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['NR_IDADE_DATA_POSSE'].dropna(), color='green', label='DEPUTADO FEDERAL')

#sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO DISTRITAL']['NR_IDADE_DATA_POSSE'].dropna(), color='red', label='DEPUTADO DISTRITAL')

sns.distplot(df_candidatos['NR_IDADE_DATA_POSSE'])

plt.title("Distribuição dos candidatos por faixa etária",color='black')

plt.grid(True,color="grey",alpha=.3)

#plt.legend(title="Cargo")

plt.xlabel('Idade na data da posse')

plt.ylabel('Distribuição')

plt.show()
# Distribuição dos candidatos por gênero

plt.figure(figsize=(15,8))

sns.countplot(x='DS_CARGO', hue='DS_GENERO', data=df_candidatos)

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de candidatos por gênero",color='black')

plt.legend(title="Sexo")

plt.xlabel('Cargo')

plt.ylabel("Quantidade de candidatos")

plt.show()



# Distribuição dos candidatos por raça

plt.figure(figsize=(10,5))

sns.countplot(x='DS_COR_RACA', data=df_candidatos)

plt.title("Quantidade de candidatos por raça",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Raça')

plt.ylabel("Quantidade")

plt.show()



# Eleitos por partido

# Criação da variável TARGET para análise de dados. CD_SIT_TOT_TURNO = 1 ELEITO, 2 ELEITO POR QP e 3 ELEITO POR MÉDIA



ELEITO = []



for i in df_candidatos['CD_SIT_TOT_TURNO']:

    if i == 1:

        ELEITO.append('sim')

    elif i == 2:

        ELEITO.append('sim')

    elif i == 3:

        ELEITO.append('sim')

    else:

        ELEITO.append('nao')

        

df_candidatos['ELEITO'] = ELEITO



#df_eleito = df_candidatos[(df_candidatos.DS_SIT_TOT_TURNO == 'ELEITO POR MÉDIA')

#                         |(df_candidatos.DS_SIT_TOT_TURNO == 'ELEITO')

#                         |(df_candidatos.DS_SIT_TOT_TURNO == 'ELEITO POR QP')]



df_eleitos = df_candidatos[df_candidatos['ELEITO'] == 'sim']

df_eleitos.info()
plt.figure(figsize=(15,18))

sns.countplot(y='NM_PARTIDO', data=df_eleitos, order = df_eleitos['NM_PARTIDO'].value_counts().index)

plt.title("Eleitos por partido",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Quantidade de candidatos eleitos')

plt.ylabel("Partido")

plt.show()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=df_candidatos['VR_DESPESA_CONTRATADA']/10, y=df_candidatos['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos votos aos gastos em campanha",color='black')

plt.xlabel('Valor das despesas em Reais (R$)')

plt.ylabel("Quantidade de votos")

plt.show()
# Custo por voto dos candidatos eleitos



eleitos = df_candidatos[df_candidatos['ELEITO'] == 'sim']

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
# Número de votos em relação à despesa Deputado Distrital

df_deputados_distritais = df_candidatos[df_candidatos['CD_CARGO'] == 8]

plt.figure(figsize=(15,5))

sns.scatterplot(x=df_deputados_distritais['VR_DESPESA_CONTRATADA']/10, y=df_deputados_distritais['QT_VOTOS_NOMINAIS'], color="magenta")

plt.grid(True, color="grey",alpha=.3)

plt.title("Distribuição dos votos aos gastos em campanha",color='black')

plt.xlabel('Valor das despesas em Reais (R$)')

plt.ylabel("Quantidade de votos")

plt.show()
df_deputados_distritais.drop(columns=['SQ_CANDIDATO', 'CD_SITUACAO_CANDIDATURA', 'CD_NACIONALIDADE', 'CD_DETALHE_SITUACAO_CAND', 'CD_CARGO'],inplace=True) 



f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(df_deputados_distritais.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=4)
# Distribuição dos candidatos por bens declarados (boxplot)

df_deputados = df_candidatos[(df_candidatos.CD_CARGO == 6) | (df_candidatos.CD_CARGO == 8)]

plt.figure(figsize=(15,10))

sns.boxplot(x='DS_CARGO', y='VR_DESPESA_CONTRATADA', data=df_deputados)

plt.title("Candidatos por faixa etária",color='black')

plt.ylabel("Idade do candidato")

plt.xlabel("Cargo")

plt.show()
# Top 10 candidatos que mais gastaram

df_candidatos[['NM_CANDIDATO','DS_CARGO','VR_DESPESA_CONTRATADA']].sort_values(by=['VR_DESPESA_CONTRATADA'], ascending=False).head(10)
# Exporta a saída

df_candidatos.to_csv('candidatos_DF.csv', sep=';', encoding='latin1', index=False)



# GERA LINK PARA BAIXAR O CSV SEM PRECISAR COMPILAR

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "candidatos_DF.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='candidatos_DF.csv')