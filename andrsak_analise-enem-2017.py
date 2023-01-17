# Localizando o dataset que será utilizado:

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importanto as bibliotecas necessárias:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected = True)
# Criando um dataset contendo apenas as colunas cujos valores serão objeto de análise:

colunas = ['CO_MUNICIPIO_RESIDENCIA', 'NO_MUNICIPIO_RESIDENCIA', 'CO_UF_RESIDENCIA',
           'SG_UF_RESIDENCIA', 'NU_IDADE', 'TP_SEXO', 'TP_ESTADO_CIVIL', 
           'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 
           'NO_MUNICIPIO_ESC', 'SG_UF_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'IN_NOME_SOCIAL',
           'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT','NU_NOTA_CN',
           'NU_NOTA_CH', 'NU_NOTA_LC','NU_NOTA_MT', 'TP_LINGUA', 'TP_STATUS_REDACAO',
           'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
           'NU_NOTA_REDACAO', 'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008',
           'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019',
           'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025', 'Q026', 'Q027']
df = pd.read_csv('/kaggle/input/enem-2017/MICRODADOS_ENEM_2017.csv', encoding='latin-1', 
                 sep=';', usecols=colunas)
# Identificando o número de linhas e colunas do dataset gerado:
df.shape
# Verificando uma amostra dos dados gerados:
df.sample(5)
# Verificando os valores válidos de cada coluna:
df.count()
# Verificando o tipo de dado contido em cada uma das colunas:
df.dtypes
# Verificando o percentual de dados faltantes:
total = df.isnull().sum().sort_values(ascending=False)
percentual = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
faltantes = pd.concat([total, percentual], axis=1, keys=['TOTAL', 'PERCENTUAL'])
faltantes
# Definindo um tamanho padrão para os gráficos:
figsize_ = (20, 12)
df_redacao_800 = df[df['NU_NOTA_REDACAO'] >= 800]
df_redacao_800
# Qual município possui mais alunos que tiraram acima de 800?

df_redacao_800['NO_MUNICIPIO_RESIDENCIA'].value_counts().sort_values(ascending = False).head(50).plot.bar(figsize = figsize_, title='Municípios com Maior Número de Alunos que Tiraram Acima de 800 na Redação')
                                                                                            
# Qual estado possui mais alunos que tiraram acima de 800?

df_redacao_800['SG_UF_RESIDENCIA'].value_counts().sort_values(ascending = False).plot.bar(figsize = figsize_, title='Número de Alunos que Tiraram Acima de 800 na Redação por UF')
# Em qual idade foi mais comum os alunos tirarem acima de 800?

df_redacao_800['NU_IDADE'].value_counts().sort_values(ascending = False).head(50).plot.bar(figsize = figsize_, title='Número de Alunos que Tiraram Acima de 800 na Redação por Idade')
# Em qual sexo é mais comum os alunos tirarem acima de 800?

df_redacao_800['TP_SEXO'].value_counts().sort_values(ascending = False).head(50).plot.bar(figsize = figsize_, title='Número de Alunos que Tiraram Acima de 800 na Redação por Sexo')





# Gerando gráfico com o número de inscritos por estados, em ordem decrescente:
df['SG_UF_RESIDENCIA'].value_counts().plot.bar(figsize = figsize_, title='Número de Inscritos por Estado')
# Gerando gráfico com o número de inscritos por idade, em ordem decrescente:
df['NU_IDADE'].value_counts().plot.bar(figsize = figsize_, title='Número de Inscritos por Idade')
# Descobrindo a média de idade dos candidados:
df['NU_IDADE'].mean()
# Criando dataframe para comparar notas e renda familiar:
df_notas_renda = df[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'Q006']].copy()
df_notas_renda
# Retirando as linhs que possuem que não possuem valores numéricos válidos:
df_notas_renda.dropna(inplace=True)
print(len(df_notas_renda))
df_notas_renda.head()
# Verificando o tipo de dado contido em cada uma das colunas:
df_notas_renda.dtypes
# Substituindo, com um dicionário, os valores referentes às faixas de renda:
df_notas_renda['Q006'].replace({'A': 'Nenhuma Renda', 'B':'Até 937,00', 
                          'C': 'De 937,01 até 1.405,50',
                          'D': 'De 1.405,51 até 1.874,00',
                          'E': 'De 1.874,01 até 2.342,50',
                          'F': 'De 2.342,51 até 2.811,00',
                          'G': 'De 2.811,01 até 3.748,00',
                          'H': 'De 3.748,01 até 4.685,00',
                          'I': 'De 4.685,01 até 5.622,00',
                          'J': 'De 5.622,01 até 6.559,00',
                          'K': 'De 6.559,01 até 7.496,00',
                          'L': 'De 7.496,01 até 8.433,00',
                          'M': 'De 8.433,01 até 9.370,00',
                          'N': 'De 9.370,01 até 11.244,00',
                          'O': 'De 11.244,01 até 14.055,00',
                          'P': 'De 14.055,01 até 18.740,00',
                          'Q': 'Mais de 18.740,00'}, inplace=True)
df_notas_renda
# Gerando gráfico com o número de inscritos por Renda, em ordem decrescente:
df_notas_renda.Q006.value_counts().plot.barh(figsize = figsize_, title='Número de Inscritos por Renda')
# Verificando o percentual de inscritos de acordo com a renda:
percentual_rendas = df_notas_renda['Q006'].value_counts().reset_index()
percentual_rendas.columns = ['Renda', 'Candidatos']
percentual_rendas['Percentual'] = (percentual_rendas['Candidatos'] / percentual_rendas['Candidatos'].sum()) * 100
percentual_rendas
# Verificando o número de alunos que tiraram 1000 em redação:
len(df[df.NU_NOTA_REDACAO == 1000])
# Verificando o números de alunos que tiraram 0 em redação:
len(df[df.NU_NOTA_REDACAO == 0])
# Verificando o número de candidados que solicitaram a utilização de seu nome social na inscrição:
len(df[df.IN_NOME_SOCIAL == 1])
# Criando histograma multi-camadas com um onda de notas, com todas as notas, exceto redação:
figsize_ = (20, 10)
fig = plt.figure(figsize=figsize_)

df_notas_renda.NU_NOTA_MT.hist(bins=200, color='red', alpha=1, label='Matemática')
plt.legend(loc='upper right')
df_notas_renda.NU_NOTA_CN.hist(bins=200, color='black', alpha=0.7, label='Ciências da Natureza')
plt.legend(loc='upper right')
df_notas_renda.NU_NOTA_CH.hist(bins=200, color='blue', alpha=0.7, label='Ciências Humanas')
plt.legend(loc='upper right')
df_notas_renda.NU_NOTA_LC.hist(bins=200, color='yellow', alpha=0.7, label='Linguagens e Códigos')
plt.legend(loc='upper right')

plt.xlabel('Notas')
plt.ylabel('Candidatos')
plt.title('Ondas de Notas')
# Verificando a distribuição de notas de Ciências da Natureza por patamar de renda familiar:

ordem_renda = ['Nenhuma Renda', 'Até 937,00', 'De 937,01 até 1.405,50',
               'De 1.405,51 até 1.874,00', 'De 1.874,01 até 2.342,50',
               'De 2.342,51 até 2.811,00', 'De 2.811,01 até 3.748,00',
               'De 3.748,01 até 4.685,00', 'De 4.685,01 até 5.622,00',
               'De 5.622,01 até 6.559,00', 'De 6.559,01 até 7.496,00',
               'De 7.496,01 até 8.433,00', 'De 8.433,01 até 9.370,00',
               'De 9.370,01 até 11.244,00', 'De 11.244,01 até 14.055,00',
               'De 14.055,01 até 18.740,00', 'Mais de 18.740,00']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x='NU_NOTA_CN',y='Q006', data=df_notas_renda, order=ordem_renda)
plt.title('Distribuição de notas de Ciências da Natureza por Patamar de Renda Familiar')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Ciências Humanas por patamar de renda familiar:

ordem_renda = ['Nenhuma Renda', 'Até 937,00', 'De 937,01 até 1.405,50',
               'De 1.405,51 até 1.874,00', 'De 1.874,01 até 2.342,50',
               'De 2.342,51 até 2.811,00', 'De 2.811,01 até 3.748,00',
               'De 3.748,01 até 4.685,00', 'De 4.685,01 até 5.622,00',
               'De 5.622,01 até 6.559,00', 'De 6.559,01 até 7.496,00',
               'De 7.496,01 até 8.433,00', 'De 8.433,01 até 9.370,00',
               'De 9.370,01 até 11.244,00', 'De 11.244,01 até 14.055,00',
               'De 14.055,01 até 18.740,00', 'Mais de 18.740,00']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x='NU_NOTA_CH',y='Q006', data=df_notas_renda, order=ordem_renda)
plt.title('Distribuição de notas de Ciências Humanas por Patamar de Renda Familiar')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Linguagens e Códigos por patamar de renda familiar:

ordem_renda = ['Nenhuma Renda', 'Até 937,00', 'De 937,01 até 1.405,50',
               'De 1.405,51 até 1.874,00', 'De 1.874,01 até 2.342,50',
               'De 2.342,51 até 2.811,00', 'De 2.811,01 até 3.748,00',
               'De 3.748,01 até 4.685,00', 'De 4.685,01 até 5.622,00',
               'De 5.622,01 até 6.559,00', 'De 6.559,01 até 7.496,00',
               'De 7.496,01 até 8.433,00', 'De 8.433,01 até 9.370,00',
               'De 9.370,01 até 11.244,00', 'De 11.244,01 até 14.055,00',
               'De 14.055,01 até 18.740,00', 'Mais de 18.740,00']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x='NU_NOTA_LC',y='Q006', data=df_notas_renda, order=ordem_renda)
plt.title('Distribuição de notas de Linguagens e Códigos por Patamar de Renda Familiar')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Matemática por patamar de renda familiar:

ordem_renda = ['Nenhuma Renda', 'Até 937,00', 'De 937,01 até 1.405,50',
               'De 1.405,51 até 1.874,00', 'De 1.874,01 até 2.342,50',
               'De 2.342,51 até 2.811,00', 'De 2.811,01 até 3.748,00',
               'De 3.748,01 até 4.685,00', 'De 4.685,01 até 5.622,00',
               'De 5.622,01 até 6.559,00', 'De 6.559,01 até 7.496,00',
               'De 7.496,01 até 8.433,00', 'De 8.433,01 até 9.370,00',
               'De 9.370,01 até 11.244,00', 'De 11.244,01 até 14.055,00',
               'De 14.055,01 até 18.740,00', 'Mais de 18.740,00']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x='NU_NOTA_MT',y='Q006', data=df_notas_renda, order=ordem_renda)
plt.title('Distribuição de notas de Matemática por Patamar de Renda Familiar')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Linguagens e Códigos por patamar de renda familiar:

ordem_renda = ['Nenhuma Renda', 'Até 937,00', 'De 937,01 até 1.405,50',
               'De 1.405,51 até 1.874,00', 'De 1.874,01 até 2.342,50',
               'De 2.342,51 até 2.811,00', 'De 2.811,01 até 3.748,00',
               'De 3.748,01 até 4.685,00', 'De 4.685,01 até 5.622,00',
               'De 5.622,01 até 6.559,00', 'De 6.559,01 até 7.496,00',
               'De 7.496,01 até 8.433,00', 'De 8.433,01 até 9.370,00',
               'De 9.370,01 até 11.244,00', 'De 11.244,01 até 14.055,00',
               'De 14.055,01 até 18.740,00', 'Mais de 18.740,00']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x='NU_NOTA_REDACAO',y='Q006', data=df_notas_renda, order=ordem_renda)
plt.title('Distribuição de notas de Redação por Patamar de Renda Familiar')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Criando gráfico com os números de inscritos por sexo:
df.TP_SEXO.value_counts().plot.bar()
# Criando DF para comparar notas e sexo:

df_notas_sexo = df[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'TP_SEXO']].copy()
df_notas_sexo.dropna(inplace=True)
print(len(df_notas_sexo))
df_notas_sexo.head()
df_notas_sexo
# Verificando a distribuição de notas de Ciências da Natureza por Sexo:

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_sexo['all'] = ''
ax = sns.violinplot(y=df_notas_sexo['all'], x=df_notas_sexo['NU_NOTA_CN'], hue=df_notas_sexo['TP_SEXO'], data=df_notas_renda, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências da Natureza por Sexo')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Ciências Humanas por Sexo:

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_sexo['all'] = ''
ax = sns.violinplot(y=df_notas_sexo['all'], x=df_notas_sexo['NU_NOTA_CH'], hue=df_notas_sexo['TP_SEXO'], data=df_notas_renda, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências Humanas por Sexo')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Linguagens e Códigos por Sexo:

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_sexo['all'] = ''
ax = sns.violinplot(y=df_notas_sexo['all'], x=df_notas_sexo['NU_NOTA_LC'], hue=df_notas_sexo['TP_SEXO'], data=df_notas_renda, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Linguagens e Códigos por Sexo')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Matemática por Sexo:

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_sexo['all'] = ''
ax = sns.violinplot(y=df_notas_sexo['all'], x=df_notas_sexo['NU_NOTA_MT'], hue=df_notas_sexo['TP_SEXO'], data=df_notas_renda, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Matemática por Sexo')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Redação por Sexo

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_sexo['all'] = ''
ax = sns.violinplot(y=df_notas_sexo['all'], x=df_notas_sexo['NU_NOTA_REDACAO'], hue=df_notas_sexo['TP_SEXO'], data=df_notas_renda, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Redação por Sexo')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando o número de alunos que possuem ou não internet em sua residência (A- não possui; B - possui)
df['Q025'].value_counts().plot.bar()
# Avaliando a influência da internet e da escola no desempenho dos candidatos:
# Criando DF para comparar notas e renda familiar

df_notas_internet_escola = df[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'Q025', 'Q027']].copy()
df_notas_internet_escola.dropna(inplace=True)
df_notas_internet_escola['Q025'].replace({'A': 'Não', 'B':'Sim'}, inplace=True)
df_notas_internet_escola['Q027'].replace({'A': 'Apenas escola pública', 
                                          'B': 'Escola pública e privada SEM bolsa integral',
                                          'C': 'Escola pública e privada COM bolsa integral',
                                          'D': 'Apenas escola privada SEM bolsa integral',
                                          'E': 'Apenas escola privada COM bolsa integral'}, inplace=True)
print(len(df_notas_internet_escola))
df_notas_internet_escola.head()
df_notas_internet_escola
# Verificando a distribuição de notas de Ciências da Natureza por tipo de escola frequentada

ordem_escolas = ['Apenas escola pública',
                 'Escola pública e privada COM bolsa integral',
                 'Escola pública e privada SEM bolsa integral',
                 'Apenas escola privada COM bolsa integral',
                 'Apenas escola privada SEM bolsa integral']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x=df_notas_internet_escola['NU_NOTA_CN'],y=df_notas_internet_escola['Q027'], data=df_notas_internet_escola, palette='colorblind', order=ordem_escolas, inner='quartile')
plt.title('Distribuição de notas de Ciências da Natureza por Tipo de Escola Frequentada')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Ciências Humanas por tipo de escola frequentada

ordem_escolas = ['Apenas escola pública',
                 'Escola pública e privada COM bolsa integral',
                 'Escola pública e privada SEM bolsa integral',
                 'Apenas escola privada COM bolsa integral',
                 'Apenas escola privada SEM bolsa integral']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x=df_notas_internet_escola['NU_NOTA_CH'],y=df_notas_internet_escola['Q027'], data=df_notas_internet_escola, palette='colorblind', order=ordem_escolas, inner='quartile')
plt.title('Distribuição de notas de Ciências Humanas por Tipo de Escola Frequentada')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Linguagens e Códigos por tipo de escola frequentada

ordem_escolas = ['Apenas escola pública',
                 'Escola pública e privada COM bolsa integral',
                 'Escola pública e privada SEM bolsa integral',
                 'Apenas escola privada COM bolsa integral',
                 'Apenas escola privada SEM bolsa integral']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x=df_notas_internet_escola['NU_NOTA_LC'],y=df_notas_internet_escola['Q027'], data=df_notas_internet_escola, palette='colorblind', order=ordem_escolas, inner='quartile')
plt.title('Distribuição de notas de Linguagens e Códigos por Tipo de Escola Frequentada')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Matemática por tipo de escola frequentada

ordem_escolas = ['Apenas escola pública',
                 'Escola pública e privada COM bolsa integral',
                 'Escola pública e privada SEM bolsa integral',
                 'Apenas escola privada COM bolsa integral',
                 'Apenas escola privada SEM bolsa integral']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x=df_notas_internet_escola['NU_NOTA_MT'],y=df_notas_internet_escola['Q027'], data=df_notas_internet_escola, palette='colorblind', order=ordem_escolas, inner='quartile')
plt.title('Distribuição de notas de Matemática por Tipo de Escola Frequentada')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Redação por tipo de escola frequentada

ordem_escolas = ['Apenas escola pública',
                 'Escola pública e privada COM bolsa integral',
                 'Escola pública e privada SEM bolsa integral',
                 'Apenas escola privada COM bolsa integral',
                 'Apenas escola privada SEM bolsa integral']

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

sns.violinplot(x=df_notas_internet_escola['NU_NOTA_REDACAO'],y=df_notas_internet_escola['Q027'], data=df_notas_internet_escola, palette='colorblind', order=ordem_escolas, inner='quartile')
plt.title('Distribuição de notas de Redação por Tipo de Escola Frequentada')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()
# Verificando a distribuição de notas de Ciências da Natureza por Acesso à internet:

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_escola['all'] = ''
ax = sns.violinplot(y=df_notas_internet_escola['all'], x=df_notas_internet_escola['NU_NOTA_CN'], hue=df_notas_internet_escola['Q025'], data=df_notas_internet_escola, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências da Natureza por Acesso à Internet')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Ciências Humanas por Acesso à internet

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_escola['all'] = ''
ax = sns.violinplot(y=df_notas_internet_escola['all'], x=df_notas_internet_escola['NU_NOTA_CH'], hue=df_notas_internet_escola['Q025'], data=df_notas_internet_escola, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências Humanas por Acesso à Internet')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Linguagens e Códigos por Acesso à internet
# Violinplot é superior ao boxplot

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_escola['all'] = ''
ax = sns.violinplot(y=df_notas_internet_escola['all'], x=df_notas_internet_escola['NU_NOTA_LC'], hue=df_notas_internet_escola['Q025'], data=df_notas_internet_escola, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Linguagens e Códigos por Acesso à Internet')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Matemática por Acesso à internet
# Violinplot é superior ao boxplot

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_escola['all'] = ''
ax = sns.violinplot(y=df_notas_internet_escola['all'], x=df_notas_internet_escola['NU_NOTA_MT'], hue=df_notas_internet_escola['Q025'], data=df_notas_internet_escola, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Matemática por Acesso à Internet')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Redação por Acesso à internet

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_escola['all'] = ''
ax = sns.violinplot(y=df_notas_internet_escola['all'], x=df_notas_internet_escola['NU_NOTA_REDACAO'], hue=df_notas_internet_escola['Q025'], data=df_notas_internet_escola, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Redação por Acesso à Internet')
ax.set_ylabel('')
ax.set_xlabel('')
# Realizando análise de diferença de desempenho no que se refere ao acesso à internete entre alunos que frequentaram apenas escola pública no ensino médio:

df_notas_internet_publica = df_notas_internet_escola[df_notas_internet_escola['Q027'] == 'Apenas escola pública'].copy()
df_notas_internet_publica
# Verificando a distribuição de notas de Ciências da Natureza por Acesso à internet - Alunos Escolas Públicas

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_publica['all'] = ''
ax = sns.violinplot(y=df_notas_internet_publica['all'], x=df_notas_internet_publica['NU_NOTA_CN'], hue=df_notas_internet_publica['Q025'], data=df_notas_internet_publica, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências da Natureza por Acesso à Internet - Alunos Escolas Públicas')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Ciências Humanas por Acesso à internet - Alunos Escolas Públicas

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_publica['all'] = ''
ax = sns.violinplot(y=df_notas_internet_publica['all'] , x=df_notas_internet_publica['NU_NOTA_CH'], hue=df_notas_internet_publica['Q025'], data=df_notas_internet_publica, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Ciências Humanas por Acesso à Internet - Alunos Escolas Públicas')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Linguagens e Códigos por Acesso à internet - Alunos Escolas Públicas

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_publica['all'] = ''
ax = sns.violinplot(y=df_notas_internet_publica['all'], x=df_notas_internet_publica['NU_NOTA_LC'], hue=df_notas_internet_publica['Q025'], data=df_notas_internet_publica, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Linguagens e Códigos por Acesso à Internet - Alunos Escolas Públicas')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Matemática por Acesso à internet - Alunos Escolas Públicas

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_publica['all'] = ''
ax = sns.violinplot(y=df_notas_internet_publica['all'], x=df_notas_internet_publica['NU_NOTA_MT'], hue=df_notas_internet_publica['Q025'], data=df_notas_internet_publica, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Matemática por Acesso à Internet - Alunos Escolas Públicas')
ax.set_ylabel('')
ax.set_xlabel('')
# Verificando a distribuição de notas de Redação por Acesso à internet - Alunos Escolas Públicas

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

df_notas_internet_publica['all'] = ''
ax = sns.violinplot(y=df_notas_internet_publica['all'], x=df_notas_internet_publica['NU_NOTA_REDACAO'], hue=df_notas_internet_publica['Q025'], data=df_notas_internet_publica, palette="Set2", split=True, scale='count', inner='quartile')
plt.title('Distribuição de notas de Redação por Acesso à Internet - Alunos Escolas Públicas')
ax.set_ylabel('')
ax.set_xlabel('')
