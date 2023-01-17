import pandas as pd
import numpy as np
from custom_libs import eneo_functions
df_2290 = eneo_functions.carrega_trafego("2290_01052016_31052016.csv")
df_2290_fx1 = df_2290[df_2290["faixa"] == 1][["data_hora", "milissegundo", "velocidade_entrada", "velocidade_saida"]].copy()
df_2290_fx1["data_hora_milli"] = df_2290_fx1["data_hora"] + pd.to_timedelta(df_2290_fx1["milissegundo"], unit='ms')
df_2290_fx1.head()
df_2290_fx1["time_diff"] = df_2290_fx1["data_hora_milli"].diff()
df_2290_fx1.head()
df_2290_fx1["velocidade_m/s"] = np.round(df_2290_fx1["velocidade_entrada"]/3.6, 2)
df_2290_fx1.head()
df_2290_fx1["time_diff_s"] = df_2290_fx1["time_diff"].dt.total_seconds()
df_2290_fx1.head()
df_2290_fx1["espacamento_metros"] = df_2290_fx1["velocidade_m/s"] * df_2290_fx1["time_diff_s"]
df_2290_fx1.head()
df_2290_fx1.set_index("data_hora_milli", inplace=True, drop=True)
df_2290_fx1_grouped = df_2290_fx1.resample("10T", label="right", closed="left")
df_2290_fx1_10min = pd.DataFrame()
df_2290_fx1_10min["espacamento_medio"] = df_2290_fx1_grouped.mean()["espacamento_metros"]
df_2290_fx1_10min.head()
df_2290_fx1_10min["densidade_veic/km"] = (1/df_2290_fx1_10min["espacamento_medio"]) * 1000
df_2290_fx1_10min.head()