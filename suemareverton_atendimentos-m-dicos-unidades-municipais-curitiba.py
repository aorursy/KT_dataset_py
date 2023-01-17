import datetime as dt
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import math

%matplotlib inline

# Formatação de 2 casas decimais
pd.options.display.float_format = '{:.2f}'.format

# Para mostrar todas as colunas ao dar um 'head'
pd.set_option('display.max_columns', 100)

sns.set()
# Importando arquivo

df = pd.read_csv(
    filepath_or_buffer = '../input/2018-08-13_Sistema_E-Saude_Medicos_-_Base_de_Dados.csv',
    sep = ';', 
    parse_dates = True,
    encoding = 'ISO-8859-1', 
    low_memory = False
) 
df.head(5)
print(df.shape[0],'linhas e',df.shape[1],'colunas')
df.info()
df['Data do Atendimento']  = pd.to_datetime(df['Data do Atendimento'], format='%d/%m/%Y %H:%M:%S')
df['Data de Nascimento']   = pd.to_datetime(df['Data de Nascimento'], format='%d/%m/%Y %H:%M:%S')
df['Data do Internamento'] = pd.to_datetime(df['Data do Internamento'], format='%d/%m/%Y %H:%M:%S')
dataInicial = df['Data do Atendimento'].min().date()
dataFinal = df['Data do Atendimento'].max().date()
print('Atendimentos entre',dataInicial.strftime('%d/%m/%Y'), 'e', dataFinal.strftime('%d/%m/%Y'))
print('Tamanho do intervalo:', (dataFinal-dataInicial).days,'dias')
# Removendo colunas que não iremos utilizar nesta análise
df.drop(
    [
        'Código do Tipo de Unidade',
        'Código da Unidade',
        'Código do Procedimento',
        'Código do CBO',
        'Qtde Prescrita Farmácia Curitibana',
        'Qtde Dispensada Farmácia Curitibana',
        'Qtde de Medicamento Não Padronizado',
        'Área de Atuação'
    ], axis=1, inplace=True)
df = df[df['Código do CID'].notnull()]
df = df.rename(columns={'Municício': 'Município'})
df['Idade'] = (df['Data do Atendimento'] - df['Data de Nascimento']).dt.days / 365
df['Idade'] = df['Idade'].astype(int)
def classificacao_etaria(idade):
    if idade < 12:
        return 'Criança'
    elif idade < 18:
        return 'Adolescente'
    elif idade < 65:
        return 'Adulto'
    else:
        return 'Idoso'
    
df['Classificação Etária'] = df['Idade'].apply(classificacao_etaria)
df['Dia da Semana Atendimento'] = df['Data do Atendimento'].dt.dayofweek
df['Dia da Semana Atendimento Descricao'] = df['Data do Atendimento'].dt.weekday_name
df['Hora Atendimento'] = df['Data do Atendimento'].dt.hour
df['Fim de Semana'] = df['Dia da Semana Atendimento Descricao'].isin(["Saturday", "Sunday"])
def turno_do_atendimento(hora):
    if hora < 6:
        return 'Madrugada'
    elif hora < 19:
        return 'Dia'
    else: 
        return 'Noite'
    
df['Turno do Atendimento'] = df['Hora Atendimento'].apply(turno_do_atendimento)
df.info()
df.head()
df_internacao = df[df['Desencadeou Internamento'] == 'Sim']
df_atendimentos_por_dia = df.groupby([df['Data do Atendimento'].dt.date, df['Tipo de Unidade']]).size().reset_index(name="Total Atendimentos")
df_internacoes_por_dia =  df_internacao.groupby([df_internacao['Data do Internamento'].dt.date]).size().reset_index(name="Total Internações")
df_atendimentos_por_dia['Data do Atendimento']  = pd.to_datetime(df_atendimentos_por_dia['Data do Atendimento'], format= '%Y-%m-%d')
df_internacoes_por_dia['Data do Internamento']  = pd.to_datetime(df_internacoes_por_dia['Data do Internamento'], format= '%Y-%m-%d')
# Convertendo de long-format para wide-format para plotagem da série temporal
df_atendimentos_por_dia_wide = df_atendimentos_por_dia.pivot(index='Data do Atendimento', columns='Tipo de Unidade', values='Total Atendimentos')

# Substituindo 'nan' por 0
df_atendimentos_por_dia_wide = df_atendimentos_por_dia_wide.fillna(0)
df_atendimentos_por_dia_wide.head()
# Definindo índice para plotagem da série temporal
df_internacoes_por_dia.set_index('Data do Internamento', inplace=True)
df_internacoes_por_dia.head()
df_atendimentos_por_dia_wide.plot(figsize=(20,10), fontsize=20, linewidth=4)
plt.xlabel('Data do Atendimento', fontsize=20);
df_internacoes_por_dia.plot(figsize=(20,10), fontsize=20, linewidth=4)
plt.xlabel('Data da Internação', fontsize=20);
sns.countplot(df['Desencadeou Internamento'])
print(df['Desencadeou Internamento'].value_counts())
df['Desencadeou Internamento'].value_counts()[1] / df['Desencadeou Internamento'].value_counts()[0]
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Encaminhamento para Atendimento Especialista?')
sns.countplot(df['Encaminhamento para Atendimento Especialista'], ax=ax[0])

# Internados
ax[1].set_title('Solicitação de Exames?')
sns.countplot(df['Solicitação de Exames'], ax=ax[1])
proporcao_curitiba = math.trunc(len(df[df['Município'] == 'CURITIBA']) / len(df) * 100)
print(proporcao_curitiba,'% são de atendimentos referentes a cidadãos curitibanos')
# Somente os 20 principais municípios
df['Município'].value_counts().reset_index(name='Atendimentos').head(20)
# Somente as 10 principais unidades
df[df['Município'] == 'COLOMBO']['Descrição da Unidade'].value_counts().reset_index(name='Atendimentos').head(10)
df_boa_vista = df[df['Descrição da Unidade'] == 'UPA BOA VISTA']
proporcao_boa_vista_curitiba = math.trunc(len(df_boa_vista[df_boa_vista['Município'] == 'CURITIBA']) / len(df_boa_vista) * 100)
print(proporcao_boa_vista_curitiba,'% de atendimentos na UPA BOA VISTA são de atendimentos referentes a cidadãos curitibanos')
# 20 principais diagnósticos
df[ (df['Município'] == 'COLOMBO') & (df['Descrição da Unidade'] == 'UPA BOA VISTA') ]['Descrição do CID'].value_counts().reset_index(name='Atendimentos').head(20)
f,ax=plt.subplots(1,2,figsize=(14,6))

# Todos
ax[0].set_title('Todos')
df['Sexo'].value_counts().plot(kind="pie", ax=ax[0])

# Internados
ax[1].set_title('Internados')
df_internacao['Sexo'].value_counts().plot(kind="pie", ax=ax[1])
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Todos')
sns.distplot(df['Idade'], bins=20, ax=ax[0]);

# Internados
ax[1].set_title('Internados')
sns.distplot(df_internacao['Idade'], bins=20, ax=ax[1]);
f,ax=plt.subplots(1,1,figsize=(14,6))
sns.boxplot(x='Tipo de Unidade', y='Idade', hue='Desencadeou Internamento', data=df);
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Todos')
sns.countplot(data=df, x = 'Classificação Etária', order=['Criança','Adolescente','Adulto','Idoso'], ax=ax[0])

ax[1].set_title('Internados')
sns.countplot(data=df_internacao, x = 'Classificação Etária', order=['Criança','Adolescente','Adulto','Idoso'], ax=ax[1])
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Todos')
sns.countplot(data=df, x = 'Turno do Atendimento', order=['Dia','Noite','Madrugada'], ax=ax[0])

ax[1].set_title('Internados')
sns.countplot(data=df_internacao, x = 'Turno do Atendimento', order=['Dia','Noite','Madrugada'], ax=ax[1])
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Todos')
sns.countplot(data=df, x = 'Tipo de Unidade', ax=ax[0])

ax[1].set_title('Internados')
sns.countplot(data=df_internacao, x = 'Tipo de Unidade', ax=ax[1])
df_tabelas_site = pd.read_html('https://pt.wikipedia.org/wiki/Classifica%C3%A7%C3%A3o_Estat%C3%ADstica_Internacional_de_Doen%C3%A7as_e_Problemas_Relacionados_com_a_Sa%C3%BAde#Codifica%C3%A7%C3%A3o', header=0)
df_tabelas_site[0]
df['Descrição do CID'].value_counts().reset_index(name='Atendimentos').head(10)
df_internacao['Descrição do CID'].value_counts().reset_index(name='Internações').head(10)
df_internacao['Código do CID'].str[0].value_counts().head(10).plot(kind='bar')
s_cid_todos = df['Código do CID'].str[0].value_counts() / len(df) * 100
s_cid_internados = df_internacao['Código do CID'].str[0].value_counts() / len(df_internacao) * 100

df_cid = pd.concat([s_cid_todos, s_cid_internados], axis=1)
df_cid.columns = ['Todos','Internados']

df_cid['Categoria'] = df_cid.index
df_cid.fillna(0, inplace=True)
df_cid
f,ax=plt.subplots(1,2,figsize=(18,5))

# Todos
ax[0].set_title('Todos')
sns.barplot(data=df_cid, x='Categoria', y='Todos', ax=ax[0])

#Internados
ax[1].set_title('Internados')
sns.barplot(data=df_cid, x='Categoria', y='Internados', ax=ax[1])
f,ax=plt.subplots(2,2,figsize=(20,16))

# Solicitação de Exames - Todos
ax[0,0].set_title('Todos')
df['Solicitação de Exames'].value_counts().plot(kind="pie", ax=ax[0,0])

# Solicitação de Exames - Internados
ax[0,1].set_title('Internados')
df_internacao['Solicitação de Exames'].value_counts().plot(kind="pie", ax=ax[0,1])

# Encaminhamento para Especialista - Todos
ax[1,0].set_title('Todos')
df['Encaminhamento para Atendimento Especialista'].value_counts().plot(kind="pie", ax=ax[1,0])

# Encaminhamento para Especialista - Internados
ax[1,1].set_title('Internados')
df_internacao['Encaminhamento para Atendimento Especialista'].value_counts().plot(kind="pie", ax=ax[1,1])
dias_semana = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

f,ax=plt.subplots(2,2,figsize=(18,12))

# Todos
ax[0,0].set_title('Todos')
sns.countplot(data=df, x = 'Dia da Semana Atendimento Descricao', order=dias_semana, ax=ax[0,0])

ax[0,1].set_title('Internados')
sns.countplot(data=df_internacao, x = 'Dia da Semana Atendimento Descricao', order=dias_semana, ax=ax[0,1])

ax[1,0].set_title('Todos')
df['Fim de Semana'].value_counts().plot(kind="pie", ax=ax[1,0])

# Encaminhamento para Especialista - Internados
ax[1,1].set_title('Internados')
df_internacao['Fim de Semana'].value_counts().plot(kind="pie", ax=ax[1,1])
df_curitiba = df[df['Município'] == 'CURITIBA']
# 20 principais bairros
df_curitiba['Bairro'].value_counts().head(20)