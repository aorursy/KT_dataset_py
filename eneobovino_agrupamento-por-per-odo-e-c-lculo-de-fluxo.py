import pandas as pd
from custom_libs import eneo_functions # Biblioteca com funções customizadas
df_2290 = eneo_functions.carrega_trafego("2290_01052016_31052016.csv")
df_2290_fx1 = df_2290[df_2290["faixa"] == 1]
df_2290_fx1.set_index("data_hora", inplace=True, drop=True)
df_2290_fx1.head()
df_2290_fx1_grouped = df_2290_fx1.resample("10T", label="right", closed="left")
# Instanciamento de um data frame vazio para receber os dados
df_2290_fx1_10min = pd.DataFrame()

# Podemos criar a coluna do data frame no momento da atribuição dos valores.
# Usaremos a coluna "numero_de_serie" para a contagem pois é garantido que nesta coluna não existem valores nulos.
df_2290_fx1_10min["veic/10min"] = df_2290_fx1_grouped["numero_de_serie"].count()
df_2290_fx1_10min.head()