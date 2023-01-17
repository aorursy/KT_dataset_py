import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
import cufflinks as cf
%matplotlib inline
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize'] = [16,8]
df_2291 = pd.read_csv("../input/2291_01052016_31052016.csv", delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int}, parse_dates=["data_hora"])
dados = df_2291.copy()
## Setando o data_hora como index do DataFrame
dados.set_index("data_hora", inplace=True, drop=True)
## Agrupando pela faixa 1 e calculando fluxo
## Agrupando...
dados_fx3 = dados[dados["faixa"]==3]
## Agrupando os dados a cada 10 min
dados_fx3 = dados_fx3.resample("10T", label="right", closed="left")
## Adicionando a coluna de Fluxo e calculando
fluxo_fx3 = pd.DataFrame()
fluxo_fx3["Fluxo"] = dados_fx3["numero_de_serie"].count()
fluxo_fx3.head()
## Calculando densidade
densidade_fx3 = df_2291[df_2291["faixa"] == 3][["data_hora", "milissegundo", "velocidade_entrada", "velocidade_saida"]].copy()
densidade_fx3["data_hora_completa"] = densidade_fx3["data_hora"] + pd.to_timedelta(densidade_fx3["milissegundo"], unit = "ms")
densidade_fx3["diff"] = densidade_fx3["data_hora_completa"].diff()
## Exibição dos dados de densidade até o momento (apenas os 5 primeiros)
densidade_fx3.head()
## A função shift() faz com ele considere os dados da linha seguinte para o calculo. Funcao parecida com o LEAD do SQL
## (caso esteja agrupado por uma coluna em especifico)
densidade_fx3["velocidade_media"] = np.round(densidade_fx3["velocidade_entrada"]/3.6, 2).shift(1)
densidade_fx3.head()
densidade_fx3["deltaT"] = densidade_fx3["diff"].dt.total_seconds()
densidade_fx3["espacamento"] = densidade_fx3["velocidade_media"] * densidade_fx3["deltaT"]
densidade_fx3.head()
densidade_fx3.set_index("data_hora_completa", inplace=True, drop=True)
densidade_fx3.head()
densidadefx3_agrupada = densidade_fx3.resample("10T", label="right", closed="left")
densidade_fx3_10 = pd.DataFrame()
densidade_fx3_10["espacamento_medio"] = densidadefx3_agrupada.mean()["espacamento"]
densidade_fx3_10.head()
densidade_fx3_10["densidade_(veic/km)"] = (1/densidade_fx3_10["espacamento_medio"]) * 1000
densidade_fx3_10.head()
x = densidade_fx3_10.index.astype('str')
y = fluxo_fx3["Fluxo"]
data = [Scatter(x=x, y=y, name="Fluxo"),
        Scatter(x=x, y=densidade_fx3["velocidade_entrada"], name="Velocidade Media", yaxis='y2')]
layout = Layout(
        title = "Fluxo x Velocidade",
        xaxis = dict(title="Data e Hora"),
        yaxis = dict(title="Fluxo"),
        yaxis2 = dict(title="Velocidade[km/h]", overlaying='y', side='right'))
fig = Figure(data = data, layout = layout)
iplot(fig)
plt.plot(fluxo_fx3["Fluxo"], color='blue')
plt.title("Fluxo")
plt.ylabel("Fluxo veic/km")
plt.xlabel("Data e Hora")
plt.show()
