import pandas as pd
import numpy as np
from custom_libs import eneo_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
import cufflinks as cf
%matplotlib inline
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize'] = [16,8]
df_2290 = eneo_functions.carrega_trafego("2290_01052016_31052016.csv")
df_2290_calculada = eneo_functions.calcula_variaveis(df_2290, "10T")
df_2290_calculada.loc['2016-05-05 00:00:00':'2016-05-05 12:00:00'].head()
plt.plot(df_2290_calculada[df_2290_calculada["faixa"] == 1].loc['2016-05-05 00:00:00':'2016-05-06 00:00:00', "fluxo"], color='red')
plt.title("Fluxo")
plt.ylabel("Fluxo [veic/10min]")
plt.xlabel("Data e Hora")
plt.show()
df_range = df_2290_calculada[df_2290_calculada["faixa"] == 1].loc['2016-05-05 00:00:00':'2016-05-06 00:00:00', ["fluxo", "densidade", "vel_media_ent"]]
x = df_range.index.astype('str')
y = df_range["fluxo"]
data = [
    Scatter(x=x, y=y, name="Fluxo"),
    Scatter(x=x, y=df_range["vel_media_ent"],name="Velocidade Media", yaxis='y2')
]
layout = Layout(
    title = "Fluxo e Velocidade",
    xaxis=dict(title="Data e Hora"),
    yaxis=dict(title="Fluxo [veic/10min]"),
    yaxis2=dict(title="Velocidade [km/h]", overlaying='y', side='right')
)
fig = Figure(data = data, layout = layout)
iplot(fig)