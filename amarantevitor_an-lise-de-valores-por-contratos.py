# Importando Bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

print(os.listdir("../input"))
# Carregando os Dados

Demandas = pd.read_csv('../input/Demandas ME.csv', sep=';')
#Analisando o tamanho dos dataframes

print('Demandas: ', Demandas.shape)
#Analisando o Head do dataframe

Demandas.head()
#Analisando o Tail do dataframe

Demandas.tail()
# Removendo as colunas que não faram parte do Estudo

Demandas.drop(columns=['Nº do Processo SEI - Abertura Demanda', 'Data de Início de Execução da Demanda', 'Data Prevista do Término da Demanda',

              'Tempo Restante', 'Nome do Projeto / Demanda', 'Instruções Complementares / Descrição', 'Serviços/Documentos a serem entregues',

              'Serviço Previsto em TR ou Contrato', 'Tempo Real de Execução', 'Data da Entrega para Aceite ou Homologação da Demanda',

              'Data da Homologação pelo Requisitante (Recusa ou Aceite)', 'Data TRP', 'Data Fis. Técnica', 'Data RCA', 'Data TRD',

              'Processo SEI - Pagamento', 'Nº da Nota Fiscal', 'Data Vencimento NF', 'Empenho Usado', 'Motivo da Glosa',

              'Data Fisc. Administrativa', 'Data Envio CEORF', 'Detalhamento da Situação (Breve Resumo do Status da OS, em especial se estiver Suspensa, Cancelada ou Atrasada)',

              'Caminho'], inplace=True)

Demandas.columns
Demandas.head()
# Removendo outras colunas que não faram parte do Estudo

Demandas.drop(columns=['Dados da Autoridade Requisitante (Nome, Área)', 'Tipo de Item'], inplace=True)

Demandas.columns
Demandas.head()
# Analisando a distribuição de frequencia do dataframe

Demandas.describe()
Demandas.info()
# Normalizando o valor

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace('.','')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace(',','.')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace('R','')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace('$','')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace('-','0')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].str.replace(' ','0')

Demandas['Valor a ser pago R$'] = Demandas['Valor a ser pago R$'].astype(float)
Demandas.info()
# Top 10 contratos com maiores valores de consumo

plt.figure(figsize=(20,10))

contratos_valoraserpago = Demandas.groupby(['Contrato'])['Valor a ser pago R$'].sum().reset_index()

contratos_valoraserpago = contratos_valoraserpago.sort_values(by='Valor a ser pago R$',ascending =False)

contratos_valoraserpago = contratos_valoraserpago.head(10)

sns.barplot(x ='Contrato',y='Valor a ser pago R$',data = contratos_valoraserpago)

plt.show()
# Consumo de todos os contratos

plt.figure(figsize=(20,10))

valoresprevisto = Demandas.groupby(['Contrato'])['Valor a ser pago R$'].sum().reset_index()

valoresprevisto = valoresprevisto.sort_values(by='Valor a ser pago R$',ascending =False)

p = sns.barplot(x ='Contrato',y='Valor a ser pago R$',data = valoresprevisto)

for i in p.get_xticklabels():

    i.set_rotation(90)

plt.show()
# Normalizando o valor

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace('.','')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace(',','.')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace('R','')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace('$','')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace('-','0')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].str.replace(' ','0')

Demandas['Preço Previsto da Demanda'] = Demandas['Preço Previsto da Demanda'].astype(float)

Demandas.info()
# Valores previsto por situação do contrato

plt.figure(figsize=(20,10))

valoresprevisto = Demandas.groupby(['Situação da Demanda'])['Preço Previsto da Demanda'].sum().reset_index()

valoresprevisto = valoresprevisto.sort_values(by='Preço Previsto da Demanda',ascending =False)

p = sns.barplot(x ='Situação da Demanda',y='Preço Previsto da Demanda',data = valoresprevisto)

for i in p.get_xticklabels():

    i.set_rotation(90)

plt.show()
# Normalizando o valor

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace('.','')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace(',','.')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace('R','')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace('$','')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace('-','0')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].str.replace(' ','0')

Demandas['Glosa em R$'] = Demandas['Glosa em R$'].astype(float)

Demandas.info()
# Valores de Glosas por contrato

plt.figure(figsize=(20,10))

glosaporcontrato = Demandas.groupby(['Contrato'])['Glosa em R$'].sum().reset_index()

glosaporcontrato = glosaporcontrato.sort_values(by='Glosa em R$',ascending =False)

glosaporcontrato = glosaporcontrato.head(10)

p = sns.barplot(x ='Contrato',y='Glosa em R$',data = glosaporcontrato)

for i in p.get_xticklabels():

    i.set_rotation(90)

plt.show()