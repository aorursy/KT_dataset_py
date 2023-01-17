# ------------------

# Imports

# ------------------

import matplotlib.pyplot as mplot

import pandas as pd

import numpy as np

import math as math
# ------------------

# Leitura do CSV

# ------------------

df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(5)
# LIMPEZA NO DATAFRAME

# -----------------------------------

# Verificação

df.isnull().sum()
def limpar_dataset(df):

    df.dropna(inplace=True)

    return df[(df[['aeronave_ano_fabricacao']] != 0).all(axis=1)]



df = limpar_dataset(df)

verif = df.isnull().sum()

verif
# Variáveis/Colunas escolhidas e classificação

# ----------------------------

classificacao = [

["aeronave_operador_categoria","Qualitativa Nominal"],

["aeronave_fabricante","Qualitativa Nominal"],

["aeronave_motor_tipo","Qualitativa Nominal"],

["aeronave_motor_quantidade","Qualitativa Ordinal"],

["aeronave_pmd_categoria","Qualitativa Ordinal"],

["aeronave_ano_fabricacao","Quantitativa Discreta"],

["aeronave_fase_operacao","Qualitativa Nominal"],

["aeronave_nivel_dano","Qualitativa Ordinal"],

["total_fatalidades","Quantitativa Discreta"],

]
# -----------------------------

# REPOSTA: QUESTÃO 1 - ITEM A

# -----------------------------

resposta = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])

resposta
df_novo = df[[

    'aeronave_operador_categoria',

    'aeronave_fabricante', 

    'aeronave_motor_tipo',

    'aeronave_motor_quantidade',

    'aeronave_pmd_categoria',

    'aeronave_ano_fabricacao',

    'aeronave_fase_operacao',

    'aeronave_nivel_dano',

    'total_fatalidades',

]]

df_novo.head(10)
operador_categoria = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

operador_categoria
fabricante = pd.DataFrame(df["aeronave_fabricante"].value_counts())

fabricante
motor_tipo = pd.DataFrame(df["aeronave_motor_tipo"].value_counts())

motor_tipo
motor_quantidade = pd.DataFrame(df["aeronave_motor_quantidade"].value_counts())

motor_quantidade
pmd_categoria = pd.DataFrame(df["aeronave_pmd_categoria"].value_counts())

pmd_categoria
fase_operacao = pd.DataFrame(df["aeronave_fase_operacao"].value_counts())

fase_operacao
nivel_dano = pd.DataFrame(df["aeronave_nivel_dano"].value_counts())

nivel_dano
# -> Categorias nas quais os operadores de vôo estão divididos

# -> Gráfico: Categorias x Quantidade de ocorrências

# -------------------------------------------

fig_opc = pd.DataFrame(df["aeronave_operador_categoria"].value_counts()).reset_index()

fig_opc = fig_opc.rename(columns={'index':'Categorias', 'aeronave_operador_categoria':'Quantidade'})

fig_opc.plot.bar(x='Categorias', y='Quantidade', label='Categorias de Operador de Aeronave', figsize=(8,5))
# -> Fabricantes de aeronves. Mostra quantos acidentes por cada fabricante ocorreram

# -> Gráfico: Fabricantes x Quantidade de ocorrências

# -------------------------------------------

fig_fabricante = pd.DataFrame(df["aeronave_fabricante"].value_counts()).reset_index().head(15)

fig_fabricante = fig_fabricante.rename(columns={'index':'Fabricantes', 'aeronave_fabricante':'Quantidade'})

fig_fabricante.plot.bar(x='Fabricantes', y='Quantidade', label='Quantidade de Aeronaves por Fabricante', figsize=(15,10))
# -> Tipo de motor utilizado pela aeronave

# -> Gráfico: Motor Tipo x Quantidade de ocorrências

# -------------------------------------------

fig_motor_tipo = pd.DataFrame(df["aeronave_motor_tipo"].value_counts()).reset_index().head(15)

fig_motor_tipo = fig_motor_tipo.rename(columns={'index':'Tipo de motor', 'aeronave_motor_tipo':'Quantidade'})

fig_motor_tipo.plot.bar(x='Tipo de motor', y='Quantidade', label='Quantidade de ocorrências para cada tipo de motor', figsize=(8,10))
# -> Quantidade de motores da aeronave

# -> Gráfico: Quantidade de motor x Quantidade de ocorrências

# -------------------------------------------

fig_motor_quantidade = pd.DataFrame(df["aeronave_motor_quantidade"].value_counts()).reset_index().head(15)

fig_motor_quantidade = fig_motor_quantidade.rename(columns={'index':'Quantidade de motor', 'aeronave_motor_quantidade':'Quantidade'})

fig_motor_quantidade.plot.bar(x='Quantidade de motor', y='Quantidade', label='Quantidade de ocorrências para a quantidade de motores da aeronave', figsize=(8,10))
# -> Categorias do peso máximo de decolagem das aeronaves

# -> Gráfico: Categoria x Quantidade de ocorrências

# -------------------------------------------

fig_pmd_cat = pd.DataFrame(df["aeronave_pmd_categoria"].value_counts()).reset_index()

fig_pmd_cat = fig_pmd_cat.rename(columns={'index':'Categoria', 'aeronave_pmd_categoria':'Quantidade'})

fig_pmd_cat.plot.bar(x='Categoria', y='Quantidade', label='Quantidade de Aeronaves por Categoria de PMD', figsize=(6,10))
# -> Ano de fabricação das aeronaves que tiveram a ocorrência

# -> Gráfico: Ano de fabricação x Quantidade de ocorrências

# -------------------------------------------

fig_ano_fabricacao = df.aeronave_ano_fabricacao.hist(bins=80, figsize=(15,12))
# -> Em que fase da operação ocorreu determinado incidente

# -> Gráfico: Fase da operação x Quantidade de ocorrências

# -------------------------------------------

fig_fase_op = pd.DataFrame(df["aeronave_fase_operacao"].value_counts()).reset_index()

fig_fase_op = fig_fase_op.rename(columns={'index':'Fase da operação', 'aeronave_fase_operacao':'Quantidade'})

fig_fase_op.plot.bar(x='Fase da operação', y='Quantidade', label='Fase da operação X Quantidade de ocorrências', figsize=(20,12))
# -> Nível de dano que teve a ocorrência

# -> Gráfico: Nível de dano x Quantidade de ocorrências

# -------------------------------------------

fig_nivel_dano = pd.DataFrame(df["aeronave_nivel_dano"].value_counts()).reset_index()

fig_nivel_dano = fig_nivel_dano.rename(columns={'index':'Nível de dano', 'aeronave_nivel_dano':'Quantidade'})

fig_nivel_dano.plot.bar(x='Nível de dano', y='Quantidade', label='Nível de dano X Quantidade de ocorrências', figsize=(8,8))
# -> Quantidade de fatalidades que ocorreram em cada ocorrência

# -> Gráfico: Quantidade de Fatalidades x Quantidade de ocorrências

# -------------------------------------------

fig_total_fat = pd.DataFrame(df["total_fatalidades"].value_counts()).reset_index()

fig_total_fat = fig_total_fat.rename(columns={'index':'Total fatalidades', 'total_fatalidades':'Quantidade'})

fig_total_fat.plot.bar(x='Total fatalidades', y='Quantidade', label='Fatalidades no vôo X Quantidade de ocorrências', figsize=(8,15))
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)