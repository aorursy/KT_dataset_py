import pandas as pd
import numpy as np
df_2290 = pd.read_csv(
    "../input/2290_01052016_31052016.csv",
    delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int
    },
    parse_dates=["data_hora"]
)
# Método utilizando 'slice' (["nome_da_coluna"])
col_data_hora = df_2290["data_hora"]

display(col_data_hora.head())

# Método utilizando acesso da coluna como um atributo/propriedade do data frame
col_data_hora_2 = df_2290.data_hora

display(col_data_hora_2.head())
cols = df_2290[["data_hora", "faixa", "velocidade_entrada"]]

cols.head()
bool_list = df_2290["faixa"] == 1

bool_list.head()
top_100 = df_2290[:100]

top_100
fx_1 = df_2290[df_2290["faixa"] == 1]

fx_1.head(10)
fx_2_vel_menor_que_10 = df_2290[(df_2290["faixa"] == 2) & (df_2290["velocidade_entrada"] <= 10)]

fx_2_vel_menor_que_10.head(10)