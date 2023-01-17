# Importando as bibliotecas que serão utilizadas na análise



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
# Carregando a base de dados



import os

print(os.listdir("../input"))
# Realizando a carga do dataframe



df = pd.read_csv('../input/RESULTADO FULL AMBULATORIAL.csv', sep=';')
# Verificando as variáveis disponíveis na base produtos, bem como apresentando o as informações do datafreme

print(df.info())

# Apresentado as 5 primeiras linhas do dataframe

df.head()

df.head().T
# Alterar o nome da coluna DATA_ATENDIMENTO para ANO_MES_ATENDIMENTO

# 1º passo - verifica os nomes das colunas

df.columns



# 2º passo - passa o novo nome das colunas para uma variável que será utilizada como auxiliar na alteração do cabeçalho



nomes = ['ANO_MES_ATENDIMENTO', 'ANO', 'FAIXA_ETARIA', 'BENEFICIARIO','VALOR_PAGO']



#Alterando o nome da coluna DATA_ATENDIMENTO



df.columns = nomes
# Apresentando o Data Frame com o nome alterado

df.head()
# Alterando o tipo de dado atual para a seguinte formatação: string, float, string, string, float



# 1º passo - tranforma os campos em string



df['BENEFICIARIO'] = df['BENEFICIARIO'].astype(str)

df['ANO_MES_ATENDIMENTO'] = df['ANO_MES_ATENDIMENTO'].astype(str)



# ler os dados da colunas BENEFICIARIO E ANO_MES_ATENDIMENTO e retorna somente os quatros primeiros dígitos, apresentando

# da esquerda para a direita



def truncar1(BENEFICIARIO):

    return BENEFICIARIO[:11]



def truncar2(ANO_MES_ATENDIMENTO):

    return ANO_MES_ATENDIMENTO[:6]



# Aplica os valores das funções 'truncar', 'truncar1' e 'trucar2' em seus respectivos 

# campos: ANO, BENEFICIÁRIO E ANO_MES_ATENDIMENTO





df['BENEFICIARIO']= df['BENEFICIARIO'].apply(truncar1)

df['ANO_MES_ATENDIMENTO']= df['ANO_MES_ATENDIMENTO'].apply(truncar2)



# Apresenta o resultado 



df.head()
# alterando o campo valor_pago, pois o mesmo encontra-se como object e não como flutuante



# 1º passo - altera a virgula por ponto

df['VALOR_PAGO'] = df['VALOR_PAGO'].str.replace(',', '.')



# 2º passo - altera os valores do tipo str para float 

df['VALOR_PAGO'] = df['VALOR_PAGO'].astype(float)
# Apresenta o resultado da conversão.

print(df.info())
# Apresenta os valores estatísticos da variávies númericas

df.describe()
# Apresentação das variaveis categoricas



df['ANO_MES_ATENDIMENTO'].value_counts()

df['FAIXA_ETARIA'].value_counts()



# Tendo em vista a grande quantidade de informação que seria apresentada por conta da variável

# "BENEFICIARIO", optou por apresentar a essa quantidade agrupada pela faixa etaria



df[['BENEFICIARIO','FAIXA_ETARIA']].groupby('FAIXA_ETARIA').count()
# Agrupa o somatorio dos valores por ANO

df[['ANO', 'VALOR_PAGO']].groupby('ANO').sum()
#Conta a quantidade de registros por ano

p = df['ANO'].value_counts()

p.plot.bar()
# Apresenta a quantidade de beneficiários por faixa etária



df[['BENEFICIARIO','FAIXA_ETARIA']].groupby('FAIXA_ETARIA').count().plot.bar()
sns.distplot(df['ANO'].value_counts(),kde=True)
sns.distplot(df['FAIXA_ETARIA'].value_counts(), kde=True)