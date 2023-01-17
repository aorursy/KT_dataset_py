import pandas as pd
import numpy as np
from custom_libs import eneo_functions
df_2290 = eneo_functions.carrega_trafego("2290_01052016_31052016.csv")
def media_harmonica(serie_de_dados):
    if len(serie_de_dados) == 0: 
        return None # Caso não existam dados no período a função retornará None
    else:
        return np.round(len(serie_de_dados)/(sum(1/serie_de_dados)),2) # Após calculada a média o valor é arredondado em duas casas decimais
df_2290_fx1 = df_2290[df_2290["faixa"] == 1].copy()
df_2290_fx1.set_index("data_hora", inplace=True, drop=True)
df_2290_fx1_grouped = df_2290_fx1.resample("10T", label="right", closed="left")
df_2290_fx1_10min = df_2290_fx1_grouped.agg(
    {
        "velocidade_entrada": media_harmonica,
        "velocidade_saida" : media_harmonica
    }
).rename(
    columns={
        "velocidade_entrada":"velocidade_media_entrada",
        "velocidade_saida":"velocidade_media_saida"
    }
)
df_2290_fx1_10min.head()
df_2290_fx1_10min = df_2290_fx1_grouped.apply(
    {
        "velocidade_entrada": media_harmonica,
        "velocidade_saida" : media_harmonica
    }
).rename(
    columns={
        "velocidade_entrada":"velocidade_media_entrada",
        "velocidade_saida":"velocidade_media_saida"
    }
)
df_2290_fx1_10min.head()