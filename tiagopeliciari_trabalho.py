# Importação de pacotes utilizados na análise

import pandas as pd

import numpy as py

import matplotlib.pyplot as plt

import seaborn as sns
# Candidatos

cand_2014 = pd.read_csv('../input/consulta_cand_2014_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)

cand_2018 = pd.read_csv('../input/consulta_cand_2018_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)



# Patrimônio

bem_cand_2014 = pd.read_csv('../input/bem_candidato_2014_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)

bem_cand_2018 = pd.read_csv('../input/bem_candidato_2018_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)



# Despesas de campanha

despesas_2014 = pd.read_csv('../input/despesas_candidatos_2014_brasil.txt', sep=';', encoding='latin1', low_memory=False)

despesas_2018 = pd.read_csv('../input/despesas_contratadas_candidatos_2018_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)



# Resultados (número de votos)

resultado_2014 = pd.read_csv('../input/votacao_candidato_munzona_2014_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)

resultado_2018 = pd.read_csv('../input/votacao_candidato_munzona_2018_BRASIL.csv', sep=';', encoding='latin1', low_memory=False)
print('Dimensões - Candidatos 2014: ', cand_2014.shape)

print('Dimensões - Candidatos 2018: ', cand_2018.shape)

print('Dimensões - Bens candidatos 2014: ', bem_cand_2014.shape)

print('Dimensões - Bens candidatos 2018: ', bem_cand_2018.shape)

print('Dimensões - Despesas Candidatos 2014: ', despesas_2014.shape)

print('Dimensões - Despesas Candidatos 2018: ', despesas_2018.shape)

print('Dimensões - Resultado Candidatos 2014: ', resultado_2014.shape)

print('Dimensões - Resultado Candidatos 2018: ', resultado_2018.shape)
cand_2014.info()
cand_2014.head()
cand_2018.info()
cand_2018.head()
bem_cand_2014.info()
bem_cand_2014.head()
bem_cand_2018.info()
bem_cand_2018.head()
despesas_2014.info()
despesas_2014.head()
despesas_2018.info()
despesas_2018.head()
resultado_2014.info()
resultado_2014.head()
resultado_2018.info()
resultado_2018.head()
# Para a análise serão mantidas apenas as colunas consideradas úteis ao estudo. 



cand_2014.drop(columns=['DT_GERACAO', 'HH_GERACAO', 'CD_TIPO_ELEICAO', 'NM_TIPO_ELEICAO', 'NR_TURNO', 'CD_ELEICAO', 

                        'DS_ELEICAO', 'DT_ELEICAO', 'SG_UE', 'NR_CANDIDATO', 'NM_URNA_CANDIDATO', 'NM_SOCIAL_CANDIDATO',

                        'NM_EMAIL', 'NR_PARTIDO', 'NM_PARTIDO', 'SQ_COLIGACAO', 

                        'DS_COMPOSICAO_COLIGACAO', 'SG_UF_NASCIMENTO', 'CD_MUNICIPIO_NASCIMENTO', 'NM_MUNICIPIO_NASCIMENTO', 

                        'DT_NASCIMENTO', 'NR_TITULO_ELEITORAL_CANDIDATO', 'NR_DESPESA_MAX_CAMPANHA', 'ST_DECLARAR_BENS',

                        'NR_PROTOCOLO_CANDIDATURA', 'NR_PROCESSO'],inplace=True) 



cand_2018.drop(columns=['DT_GERACAO', 'HH_GERACAO', 'CD_TIPO_ELEICAO', 'NM_TIPO_ELEICAO', 'NR_TURNO', 'CD_ELEICAO', 

                        'DS_ELEICAO', 'DT_ELEICAO', 'SG_UE', 'NR_CANDIDATO', 'NM_URNA_CANDIDATO', 'NM_SOCIAL_CANDIDATO',

                        'NM_EMAIL', 'NR_PARTIDO', 'NM_PARTIDO', 'SQ_COLIGACAO', 

                        'DS_COMPOSICAO_COLIGACAO', 'SG_UF_NASCIMENTO', 'CD_MUNICIPIO_NASCIMENTO', 'NM_MUNICIPIO_NASCIMENTO', 

                        'DT_NASCIMENTO', 'NR_TITULO_ELEITORAL_CANDIDATO', 'NR_DESPESA_MAX_CAMPANHA', 'ST_DECLARAR_BENS',

                        'NR_PROTOCOLO_CANDIDATURA', 'NR_PROCESSO'],inplace=True) 
cand_2014.columns
cand_2018.columns
# Na base de dados, candidatos que vão para o segundo turno são registrados duas vezes. É necessário excluir este registro afim 

# de se evitar duplicidades, inconsistências (CD_SIT_TOT_TURNO = 6)



cand_2014 = cand_2014[cand_2014.CD_SIT_TOT_TURNO != 6]

cand_2018 = cand_2018[cand_2018.CD_SIT_TOT_TURNO != 6]



# Serão mantidos apenas os candidatos com candidatura APTA

cand_2014 = cand_2014[cand_2014.CD_SITUACAO_CANDIDATURA == 12]

cand_2018 = cand_2018[cand_2018.CD_SITUACAO_CANDIDATURA == 12]



# Outros filtros podem ser aplicados aqui
# Formatação dos dados do campo VR_BEM_CANDIDATO. Os dados em formato numérico nacional podem apresentar 

# conflitos com o padrão americano.

bem_cand_2014['VR_BEM_CANDIDATO'] = bem_cand_2014['VR_BEM_CANDIDATO'].str.replace(',','.').astype(float)
bem_cand_2018['VR_BEM_CANDIDATO'] = bem_cand_2018['VR_BEM_CANDIDATO'].str.replace(',','.').astype(float)
# Agregação dos bens dos candidatos por SQ_CANDIDATO

bem_2014 = bem_cand_2014.groupby('SQ_CANDIDATO').sum()

bem_2014.drop(columns=['ANO_ELEICAO','CD_TIPO_ELEICAO','CD_ELEICAO','NR_ORDEM_CANDIDATO','CD_TIPO_BEM_CANDIDATO'],inplace=True)
bem_2018 = bem_cand_2018.groupby('SQ_CANDIDATO').sum()

bem_2018.drop(columns=['ANO_ELEICAO','CD_TIPO_ELEICAO','CD_ELEICAO','NR_ORDEM_CANDIDATO','CD_TIPO_BEM_CANDIDATO'],inplace=True)
bem_2014.head()
bem_2018.head()
# Formatação dos dados do campo VR_BEM_CANDIDATO. Os dados em formato numérico nacional podem apresentar conflitos com o 

# padrão americano.

despesas_2014['Valor despesa'] = despesas_2014['Valor despesa'].str.replace(',','.').astype(float)
despesas_2018['VR_DESPESA_CONTRATADA'] = despesas_2018['VR_DESPESA_CONTRATADA'].str.replace(',','.').astype(float)
# Ajustes dos nomes da base 2014

novo_nome_coluna = ['Cód. Eleição', 'Desc. Eleição', 'Data e hora', 'CNPJ Prestador Conta', 'SQ_CANDIDATO', 'UF', 

                    'Sigla  Partido', 'Número candidato', 'Cargo', 'Nome candidato', 'CPF do candidato', 'Tipo do documento', 

                    'Número do documento', 'CPF/CNPJ do fornecedor', 'Nome do fornecedor', 

                    'Nome do fornecedor (Receita Federal)', 'Cod setor econômico do fornecedor', 

                    'Setor econômico do fornecedor', 'Data da despesa', 'VR_DESPESA_CONTRATADA', 'Tipo despesa', 

                    'Descriçao da despesa']

despesas_2014.columns = novo_nome_coluna

despesas_2014.columns
# Agrupamento das despesas por candidato

despesas_2014 = despesas_2014.groupby('SQ_CANDIDATO').sum()
despesas_2018 = despesas_2018.groupby('SQ_CANDIDATO').sum()
# Exclusão das colunas não utilizadas na análise

despesas_2014.drop(columns=['Cód. Eleição','CNPJ Prestador Conta', 'Número candidato', 'CPF do candidato'], inplace=True)
despesas_2018.drop(columns=['ANO_ELEICAO', 'CD_TIPO_ELEICAO', 'CD_ELEICAO', 'ST_TURNO', 'SQ_PRESTADOR_CONTAS', 

                            'NR_CNPJ_PRESTADOR_CONTA', 'CD_CARGO', 'NR_CANDIDATO', 'NR_CPF_CANDIDATO', 

                            'NR_CPF_VICE_CANDIDATO', 'NR_PARTIDO', 'CD_TIPO_FORNECEDOR', 'CD_CNAE_FORNECEDOR', 

                            'NR_CPF_CNPJ_FORNECEDOR', 'CD_ESFERA_PART_FORNECEDOR', 'CD_MUNICIPIO_FORNECEDOR', 

                            'SQ_CANDIDATO_FORNECEDOR', 'NR_CANDIDATO_FORNECEDOR', 'CD_CARGO_FORNECEDOR', 

                            'NR_PARTIDO_FORNECEDOR', 'CD_ORIGEM_DESPESA', 'SQ_DESPESA'], inplace=True)
despesas_2014.head()
despesas_2018.head()
cand_2014 = pd.merge(cand_2014, bem_2014, on='SQ_CANDIDATO', how='left')

cand_2014 = pd.merge(cand_2014, despesas_2014, on='SQ_CANDIDATO', how='left')
cand_2018 = pd.merge(cand_2018, bem_2018, on='SQ_CANDIDATO', how='left')

cand_2018 = pd.merge(cand_2018, despesas_2018, on='SQ_CANDIDATO', how='left')
cand_2014 = pd.merge(cand_2014, resultado_2014, on='SQ_CANDIDATO', how='left')

cand_2018 = pd.merge(cand_2018, resultado_2018, on='SQ_CANDIDATO', how='left')
# União das duas bases

cand = cand_2014.append(cand_2018)

cand.shape
# Criação da variável TARGET para análise de dados. CD_SIT_TOT_TURNO = 1 ELEITO, 2 ELEITO POR QP e 3 ELEITO POR MÉDIA



ELEITO = []



for i in cand['CD_SIT_TOT_TURNO']:

    if i == 1:

        ELEITO.append('sim')

    elif i == 2:

        ELEITO.append('sim')

    elif i == 3:

        ELEITO.append('sim')

    else:

        ELEITO.append('nao')

        

cand['ELEITO'] = ELEITO

################################# A base "cand" será utilizada para a realização das análises ###############################

#cand.head()



# Setando apenas deputados federais / estaduais

# ESTADUAL CD_CARGO = 7

# DISTRITAL CD_CARGO = 8

# FEDERAL CD_CARGO = 6



cand = cand[(cand.CD_CARGO == 6) | (cand.CD_CARGO == 7) | (cand.CD_CARGO == 8)]

cand.head()
# Maiores patrimônios

cand.nlargest(5, 'VR_BEM_CANDIDATO')
# Maiores gastos na campanha

cand.nlargest(5, 'VR_DESPESA_CONTRATADA')
# candidatos mais velhos

cand.nlargest(5, 'NR_IDADE_DATA_POSSE')
# candidatos mais novos



cand.nsmallest(5, 'NR_IDADE_DATA_POSSE')
# Correlação entre os dados

cand.corr()
# Correlação

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(cand.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=4)
# Distribuição dos candidatos por faixa etária

plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO ESTADUAL']['NR_IDADE_DATA_POSSE'].dropna(), color='blue', label='DEPUTADO ESTADUAL')

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO FEDERAL']['NR_IDADE_DATA_POSSE'].dropna(), color='green', label='DEPUTADO FEDERAL')

sns.distplot(cand[cand['DS_CARGO'] == 'DEPUTADO DISTRITAL']['NR_IDADE_DATA_POSSE'].dropna(), color='red', label='DEPUTADO DISTRITAL')

plt.title("Distribuição dos candidatos por faixa etária",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Cargo")

plt.xlabel('Idade na data da posse')

plt.ylabel('Quantidade de candidatos')

plt.show()
# Número de votos em relação à idade

plt.figure(figsize=(15,5))

sns.scatterplot(x='NR_IDADE_DATA_POSSE', y='TOTAL_VOTOS', data=cand, color="red")

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos votos em relação à idade",color='black')

plt.xlabel('Idade na data de posse')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos candidatos por faixa etária (boxplot)

plt.figure(figsize=(10,5))

sns.boxplot(x='DS_CARGO', y='NR_IDADE_DATA_POSSE', data=cand)

plt.title("Candidatos por faixa etária",color='black')

plt.ylabel("Idade do candidato")

plt.xlabel("Cargo")

plt.show()
# Número de votos em relação à despesa

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand['VR_DESPESA_CONTRATADA']/10, y=cand['TOTAL_VOTOS'], color="magenta")

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos votos aos gastos em campanha",color='black')

plt.xlabel('Valor das despesas em Reais (R$)')

plt.ylabel("Quantidade de votos")

plt.show()
# Número de votos em relação aos bens do candidato

plt.figure(figsize=(15,5))

sns.scatterplot(x=cand['VR_BEM_CANDIDATO']/100000000, y=cand['TOTAL_VOTOS'], data=cand, color="blue")

plt.grid(True,color="grey",alpha=.3)

plt.title("Distribuição dos votos aos bens do candidato",color='black')

plt.xlabel('Valor de bens declarados em milhões de reais')

plt.ylabel("Total de Votos")

plt.show()
# Distribuição dos candidatos por gênero

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_GENERO', data=cand)

plt.grid(True,color="grey",alpha=.3)

plt.title("Quantidade de candidatos por gênero",color='black')

plt.legend(title="Sexo")

plt.xlabel('Cargo')

plt.ylabel("Quantidade de candidatos")

plt.show()
# Total de votos em relação ao gênero do candidato

plt.figure(figsize=(5,10))

sns.boxplot(x='DS_GENERO', y='TOTAL_VOTOS', data=cand)

plt.title("Votação por gênero",color='black')

plt.xlabel('Gênero')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos votos por gênero

plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_GENERO'] == 'MASCULINO']['TOTAL_VOTOS'].dropna(), color='blue', label='M')

sns.distplot(cand[cand['DS_GENERO'] == 'FEMININO']['TOTAL_VOTOS'].dropna(), color='pink', label='F')

plt.title("Distribuição dos candidatos por gênero",color='black')

plt.legend(title="Sexo do Candidato")

plt.xlabel('Quantidade de Votos')

plt.grid(True,color="grey",alpha=.3)

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

sns.boxplot(x='DS_GRAU_INSTRUCAO', y='TOTAL_VOTOS', data=cand)

plt.xticks(rotation=90)

plt.title("Votação por formação acadêmica",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.xlabel('Grau de instrução')

plt.ylabel("Quantidade de votos")

plt.show()
# Distribuição dos votos por formação acadêmica

plt.figure(figsize=(15,5))

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'SUPERIOR COMPLETO']['TOTAL_VOTOS'].dropna(), color='blue', label='SUPERIOR COMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'SUPERIOR INCOMPLETO']['TOTAL_VOTOS'].dropna(), color='yellow', label='SUPERIOR INCOMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'ENSINO FUNDAMENTAL COMPLETO']['TOTAL_VOTOS'].dropna(), color='green', label='FUNDAMENTAL COMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'ENSINO FUNDAMENTAL INCOMPLETO']['TOTAL_VOTOS'].dropna(), color='purple', label='FUNDAMENTAL INCOMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'ENSINO MÉDIO COMPLETO']['TOTAL_VOTOS'].dropna(), color='gray', label='MEDIO COMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'ENSINO MÉDIO INCOMPLETO']['TOTAL_VOTOS'].dropna(), color='pink', label='MEDIO INCOMPLETO')

sns.distplot(cand[cand['DS_GRAU_INSTRUCAO'] == 'LÊ E ESCREVE']['TOTAL_VOTOS'].dropna(), color='cyan', label='LE E ESCREVE')

plt.title("Distribuição dos candidatos por formação acadêmica",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Grau de Instrução")

plt.xlabel('Quantidade de Votos')

plt.show()
# Distribuição dos candidatos por raça

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_COR_RACA', data=cand)

plt.title("Quantidade de candidatos por raça",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Raça")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
# Distribuição dos candidatos por estado civil

plt.figure(figsize=(10,5))

sns.countplot(x='DS_CARGO', hue='DS_ESTADO_CIVIL', data=cand)

plt.title("Quantidade de candidatos estado civil",color='black')

plt.grid(True,color="grey",alpha=.3)

plt.legend(title="Estado Civil")

plt.xlabel('Cargo')

plt.ylabel("Quantidade")

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



eleitos = cand[cand['ELEITO'] == 'sim']

eleitos = eleitos[['ANO_ELEICAO', 'SG_UF', 'DS_CARGO', 'VR_DESPESA_CONTRATADA', 'TOTAL_VOTOS']]

eleitos = eleitos.groupby(['ANO_ELEICAO', 'SG_UF', 'DS_CARGO']).sum().reset_index()

eleitos['CUSTO_VOTO'] = eleitos['VR_DESPESA_CONTRATADA'] / eleitos['TOTAL_VOTOS']



eleitos



# Distribuição dos candidatos por custo voto

plt.figure(figsize=(15,5))

sns.barplot(x='SG_UF', y='CUSTO_VOTO', hue='ANO_ELEICAO', data=eleitos, ci=None)

plt.xlabel('Unaidade da Federação')

plt.ylabel("Custo por voto em Reias (R$)")

plt.title("Custo por voto para os eleitos")

plt.legend(title="Ano Eleição")

plt.axhline(y=eleitos['CUSTO_VOTO'].mean())

cand2 = cand
mask = (cand['DS_ESTADO_CIVIL'] == "SEPARADO(A) JUDICIALMENTE") | (cand['DS_ESTADO_CIVIL'] == "DIVORCIADO(A)")

cand['DS_ESTADO_CIVIL'] = cand['DS_ESTADO_CIVIL'].mask(mask, "SEPARADO")



mask = (cand['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL INCOMPLETO") | (cand['DS_GRAU_INSTRUCAO'] == "ENSINO FUNDAMENTAL COMPLETO")| (cand['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO INCOMPLETO")

cand['DS_GRAU_INSTRUCAO'] = cand['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO FUNDAMENTAL")



mask = (cand['DS_GRAU_INSTRUCAO'] == "ENSINO MÉDIO COMPLETO") | (cand['DS_GRAU_INSTRUCAO'] == "SUPERIOR INCOMPLETO")

cand['DS_GRAU_INSTRUCAO'] = cand['DS_GRAU_INSTRUCAO'].mask(mask, "ENSINO MÉDIO")



#ELEITORES = {

#    "AC": 547692,

#    "AL": 2187947,

#    "AM": 2428009,

#    "AP": 512074,

#    "BA": 10393166,

#    "CE": 6344479,

#    "DF": 2084379,

#    "ES": 2754775,

#    "GO": 4454079,

#    "MA": 4537180,

#    "MG": 15701164,

#    "MS": 1878017,

#    "MT": 2330175,

#    "PA": 5499285,

#    "PB": 2867625,

#    "PE": 6570060,

#    "PI": 2370886,

#    "PR": 7971350,

#    "RJ": 12409201,

#    "RN": 2373586,

#    "RO": 1175844,

#    "RR": 333464,

#    "RS": 8354864,

#    "SC": 5070322,

#    "SE": 1577039,

#    "SP": 33028916,

#    "TO": 1039439

#}



ELEITORES = []



for i in cand['SG_UF']:

    if i == "AC":

        ELEITORES.append(547692)

    elif i == "AL":

        ELEITORES.append(2187947)

    elif i == "AM":

        ELEITORES.append(2428009)

    elif i == "AP":

        ELEITORES.append(512074)

    elif i == "BA":

        ELEITORES.append(10393166)

    elif i == "CE":

        ELEITORES.append(6344479)

    elif i == "DF":

        ELEITORES.append(2084379)

    elif i == "ES":

        ELEITORES.append(2754775)

    elif i == "GO":

        ELEITORES.append(4454079)

    elif i == "MA":

        ELEITORES.append(4537180)

    elif i == "MG":

        ELEITORES.append(15701164)

    elif i == "MS":

        ELEITORES.append(1878017)

    elif i == "MT":

        ELEITORES.append(2330175)

    elif i == "PA":

        ELEITORES.append(5499285)

    elif i == "PB":

        ELEITORES.append(2867625)

    elif i == "PE":

        ELEITORES.append(6570060)

    elif i == "PI":

        ELEITORES.append(2370886)

    elif i == "PR":

        ELEITORES.append(7971350)    

    elif i == "RJ":

        ELEITORES.append(12409201)

    elif i == "RN":

        ELEITORES.append(2373586)

    elif i == "RO":

        ELEITORES.append(1175844)

    elif i == "RR":

        ELEITORES.append(333464)

    elif i == "RS":

        ELEITORES.append(8354864)

    elif i == "SC":

        ELEITORES.append(5070322)

    elif i == "SE":

        ELEITORES.append(1577039)

    elif i == "SP":

        ELEITORES.append(33028916)    

    else:    

        ELEITORES.append(1039439)

        

cand['ELEITORES'] = ELEITORES

cand.to_csv('cand_final.csv', sep=';', encoding='latin1', index=False)
# GERA LINK PARA BAIXAR O CSV SEM PRECISAR COMPILAR

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "cand_final.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='cand_final.csv')