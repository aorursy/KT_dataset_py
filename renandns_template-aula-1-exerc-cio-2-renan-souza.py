import pandas as pd

import numpy as np

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(5)
resposta = [["aeronave_operador_categoria","Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_motor_quantidade","Qualitativa Nominal"],

            ["aeronave_assentos","Quantitativa Discreta"],

            ["aeronave_pais_fabricante","Qualitativa Nominal"],

            ["aeronave_fase_operacao","Qualitativa Nominal"],

            ["total_fatalidades","Quantitativa Discreta"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df[resposta["Variavel"]]
var1 = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

var1
var2 = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

var2.index

#arr2 = np.array(var2)

#var2

#arr2
var3 = pd.DataFrame(df["aeronave_motor_quantidade"].value_counts())

var3
var4 = pd.DataFrame(df["aeronave_pais_fabricante"].value_counts())

var4
var5 = pd.DataFrame(pd.DataFrame(df["aeronave_fase_operacao"].value_counts()))

var5
df["aeronave_operador_categoria"].value_counts().plot(kind='bar', title='Categoria Operador - Completo')
df["aeronave_operador_categoria"].value_counts().head(3).plot(kind='pie', title='Categoria Operador - Top3')
df["aeronave_tipo_veiculo"].value_counts().plot(kind='bar', title='Tipo de Aeronave - Completo')
df["aeronave_tipo_veiculo"].value_counts().head(3).plot(kind='pie', title='Tipo de Aeronave - Top 3')
df["aeronave_motor_quantidade"].value_counts().plot(kind='bar', title='Quantidade Motor')
df["aeronave_fase_operacao"].value_counts().plot(kind='bar', title='Fase de Operação - Completo')
df["aeronave_fase_operacao"].value_counts().head(10).plot(kind='bar', title='Fase de Operação - Top 10')
df["aeronave_pais_fabricante"].value_counts().plot(kind='bar', title='Pais Fabricante - Completo')
df["aeronave_pais_fabricante"].value_counts().head(5).plot(kind='bar', title='Pais Fabricante - Top 5')