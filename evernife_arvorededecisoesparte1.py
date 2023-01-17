import time

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas import Series, DataFrame

import plotly.express as px

#Ultimos 10 dias  de morte no BRASIL!

original_df = pd.read_csv("../input/cases-brazil-states-18-05-2020.csv");

df_br = original_df.filter(['date', 'state', 'deaths']);

df_br = df_br.loc[df_br['state'] == "TOTAL"];



print("Ultimos 10 dias de mortes no Brasil")

print(df_br.tail(10));

print("\n\nGráfico da curva de crescimento das mortes")



fig = px.bar(df_br, x="date", y="deaths", color="deaths", barmode="group")

fig.show()
df = original_df.loc[original_df['date'] == "2020-05-18"];

df = df.loc[df['state'] != "TOTAL"];

df = df.loc[df['deaths'] > 100];

df = df.filter(['state', 'deaths']);

df = df.sort_values(by=['deaths'],ascending=False)



print("\n\nRelação de mortes por estados com mais de 100 mortes")

fig = px.bar(df, x="state", y="deaths", color="state", barmode="group", width=1000)

fig.show()
#
#Ultimos 10 dias  de morte no Estado DE SÃO PAULO!

df_sp = original_df.filter(['date', 'state', 'deaths']);

df_sp = df_sp.loc[df_sp['state'] == "SP"];



print("Ultimos 10 dias de Mortes no Estado de São Paulo")

print(df_sp.tail(10));



print("\n\nGráfico da curva de crescimento das mortes")



#foundNull = df_sp['deaths'].isnull().values.any(); #Nenhum valor nulo encontrado





fig = px.bar(df_sp, x="date", y="deaths", color="deaths", barmode="group")

fig.show()