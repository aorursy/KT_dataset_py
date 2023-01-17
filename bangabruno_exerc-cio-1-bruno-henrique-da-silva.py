import pandas as pd

import numpy as np



df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(10)
import pandas as pd

variaveis = [["aeronave_operador_categoria", "Qualitativa Nominal"], 

             ["aeronave_tipo_veiculo", "Qualitativa Nominal"], 

             ["aeronave_motor_tipo", "Qualitativa Nominal"], 

             ["aeronave_motor_quantidade", "Qualitativa Ordinal"], 

             ["aeronave_pmd_categoria", "Qualitativa Ordinal"],

             ["aeronave_assentos", "Quantitativa Discreta"], 

             ["aeronave_ano_fabricacao", "Qualitativa Ordinal"], 

             ["aeronave_registro_segmento", "Qualitativa Nominal"], 

             ["total_fatalidades", "Quantitativa Discreta"]]



resposta = pd.DataFrame(variaveis, columns=["Variável", "Classificação"])

resposta
variaveis

print(df["aeronave_operador_categoria"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_tipo_veiculo"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_motor_tipo"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_motor_quantidade"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_pmd_categoria"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_ano_fabricacao"].value_counts())

print("============================================")

print("--------------------------------------------")

print("============================================")

print(df["aeronave_registro_segmento"].value_counts())
import pandas as pd

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



qt_fatalidades = go.Bar(x = np.sort(df.aeronave_operador_categoria.unique()),

                     y = df.groupby(["aeronave_operador_categoria"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por categoria',

                   yaxis={'title':'Quantidade'},

                   xaxis={'title': 'Categoria'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)
qt_fatalidades = go.Bar(x = np.sort(df.aeronave_tipo_veiculo.unique()),

                     y = df.groupby(["aeronave_tipo_veiculo"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32", "#e84118", "#e1b12c", "#273c75", "#40739e"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por tipo de veículo',

                   yaxis={'title':'Quantidade'},

                   xaxis={'title': 'Tipo de veículo'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)
qt_fatalidades = go.Bar(x = np.sort(df.aeronave_motor_tipo.unique()),

                     y = df.groupby(["aeronave_motor_tipo"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32", "#e84118", "#e1b12c", "#273c75", "#40739e"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por tipo de motor',

                   yaxis={'title':'Quantidade'},

                   xaxis={'title': 'Tipo de motor'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)
qt_fatalidades = go.Bar(x = np.sort(df.aeronave_motor_quantidade.unique()),

                     y = df.groupby(["aeronave_motor_quantidade"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32", "#e84118", "#e1b12c", "#273c75", "#40739e"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por qtd. de motor',

                   yaxis={'title':'Quantidade de fatalidades'},

                   xaxis={'title': 'Qtd. de motor'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)
qt_fatalidades = go.Bar(x = np.sort(df.aeronave_pmd_categoria.unique()),

                     y = df.groupby(["aeronave_pmd_categoria"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32", "#e84118", "#e1b12c", "#273c75", "#40739e"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por categoria PMD',

                   yaxis={'title':'Quantidade'},

                   xaxis={'title': 'Categoria PMD'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)
# Considerando apenas aeronaves com ano de fabricação preenchido

df_ano = df[df["aeronave_ano_fabricacao"] > 0]

trace = go.Scatter(x = np.sort(df_ano["aeronave_ano_fabricacao"].unique()),

                    y = df_ano.groupby("aeronave_ano_fabricacao").sum()["total_fatalidades"],

                    mode = 'lines',

                    name = 'Gráfico com linha pontilhada',

                    line = {

                            'dash': 'dot'})

data = [trace]



py.iplot(data)
qt_fatalidades = go.Bar(x = np.sort(df.aeronave_registro_segmento.unique()),

                     y = df.groupby(["aeronave_registro_segmento"]).sum()["total_fatalidades"],

                     name = 'Total de fatalidades',

                     marker = {"color": ["#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#9c88ff", "#44bd32", "#e84118", 

                                         "#e1b12c", "#273c75", "#40739e", "#0fbcf9", "#ffa801", "#485460"]})



data = [qt_fatalidades]



# Criando Layout

layout = go.Layout(title='Quantidade de fatalidades por registro de segmento',

                   yaxis={'title':'Quantidade'},

                   xaxis={'title': 'Segmento'})



# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)



# Exibindo figura/gráfico

py.iplot(fig)