# Imports

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import math as math
# Reading csv file

def read_csv():

    df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

    return df
def clean_df(df):

    df.dropna(inplace=True)

    return df[(df[['aeronave_ano_fabricacao']] != 0).all(axis=1)]
df = read_csv()

df.head()
empty = df.isnull().sum()

empty
df = clean_df(df)

empty = df.isnull().sum()

empty
df[[

    'aeronave_operador_categoria',

    'aeronave_fabricante', 

    'aeronave_pmd',  

    'aeronave_pmd_categoria', 

    'aeronave_assentos', 

    'aeronave_ano_fabricacao',

    'total_fatalidades',

]].head()
resp = [

    ["aeronave_operador_categoria", "Qualitativa Nominal"],

    ["aeronave_fabricante","Qualitativa Nominal"],

    ["aeronave_pmd","Quantitativa Discreta"],

    ["aeronave_pmd_categoria","Qualitativa Ordinal"],

    ["aeronave_assentos","Quantitativa Discreta"],

    ["aeronave_ano_fabricacao","Quantitativa Discreta"],

    ["total_fatalidades","Quantitativa Discreta"],

]



resp = pd.DataFrame(resp, columns=["Variavel", "Classificação"])



resp
op_cat = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

op_cat
fab = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

fab
aero_cat = pd.DataFrame(df["aeronave_pmd_categoria"].value_counts())

aero_cat
fig = pd.DataFrame(df["aeronave_operador_categoria"].value_counts()).reset_index()

fig = fig.rename(columns={'index':'Categorias', 'aeronave_operador_categoria':'Quantidade'})

fig.plot.bar(x='Categorias', y='Quantidade', label='Categorias de Operador de Aeronave', figsize=(7,5))
fig = pd.DataFrame(df["aeronave_fabricante"].value_counts()).reset_index().head(15)

fig = fig.rename(columns={'index':'Fabricantes', 'aeronave_fabricante':'Quantidade'})

fig.plot.bar(x='Fabricantes', y='Quantidade', label='Quantidade de Aeronaves por Fabricante', figsize=(12,10))
fig = pd.DataFrame(df["aeronave_pmd_categoria"].value_counts()).reset_index()

fig = fig.rename(columns={'index':'Categoria', 'aeronave_pmd_categoria':'Quantidade'})

fig.plot.bar(x='Categoria', y='Quantidade', label='Quantidade de Aeronaves por Categoria de PMD', figsize=(12,10))
df.aeronave_pmd.hist(figsize=(12,10))
df.aeronave_assentos.hist(bins=15, figsize=(12,10))
df.aeronave_ano_fabricacao.hist(bins=40, figsize=(12,10))
fig = pd.DataFrame(df["total_fatalidades"].value_counts()).reset_index()

fig = fig.rename(columns={'index':'Fatalidades', 'total_fatalidades':'Total_Ocorrencias'})

fig.plot.bar(x='Fatalidades', y='Total_Ocorrencias', label='Quantidade de Fatalidades por Ocorrencia', figsize=(12,10))