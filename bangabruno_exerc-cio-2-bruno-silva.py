import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv("../input/dataviz-facens-20182-ex3/BlackFriday.csv")
df.head()
sbn.set(style="whitegrid")
plt.figure(figsize=(14, 8))
plt.title('Qtd. de Compras x Faixa Etária')
sbn.violinplot(x = np.sort(df["Age"]), y = df["Purchase"], scale = "count")
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

df_custom = pd.DataFrame(df.groupby("Product_ID").count()["Purchase"].sort_values(ascending=False).head(10))
df_custom = df_custom.reset_index()

produtos = go.Bar(x = df_custom["Product_ID"],
                    y = df_custom["Purchase"],
                    name = "Produtos",
                    marker = {'color': ['#FF5722', '#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#4CAF50', '#FF9800', '#8BC34A', '#795548']},
                    opacity = .6)

data = [produtos]

# Criando Layout
layout = go.Layout(title='Os 10 Produtos mais comprados')

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)

# Exibindo figura/gráfico
py.iplot(fig)
#Obtem as 5 ocupações mais frequentes
ocupacoes = pd.DataFrame(df.Occupation.value_counts().sort_values(ascending=False).head(5)).reset_index()
ocupacoes.rename(columns={"index": "Codigo", "Occupation": "Qtd"}, inplace=True)
ocupacoes

data = []

for o in ocupacoes["Codigo"].sort_values(ascending=True):
    df_ocupacoes = df[df["Occupation"] == o]
    
    trace = go.Bar(x = np.sort(df_ocupacoes["Age"].unique()),
                   y = df_ocupacoes.groupby("Age").sum()["Purchase"],
                   name = "Ocupação " + str(o),
                   opacity = .5)

    data.append(trace)

# Criando Layout
layout = go.Layout(title='Distribuição dos valores gastos por faixa etária para as 5 ocupações mais frequentes')

# Criando figura que será exibida
fig = go.Figure(data = data, layout=layout)

# Exibindo figura/gráfico
py.iplot(fig)
df_custom = df[df["Purchase"] > 9000]

df_married = df_custom[df_custom["Marital_Status"] == 1]
df_single = df_custom[df_custom["Marital_Status"] == 0]

data = []
    
married = go.Bar(x = df_married["Occupation"].unique(),
               y = df_married.groupby("Occupation").sum()["Purchase"].sort_values(ascending=True),
               name = "Casados",
               opacity = .5)

single = go.Bar(x = df_single["Occupation"].unique(),
               y = df_single.groupby("Occupation").sum()["Purchase"].sort_values(ascending=True),
               name = "Não casados",
               opacity = .5)

data = [married, single]

# Criando Layout
layout = go.Layout(title='Distribuição dos valores entre Ocupação e Estado Civil para compras > 9000')

# Criando figura que será exibida
fig = go.Figure(data = data, layout=layout)

# Exibindo figura/gráfico
py.iplot(fig)