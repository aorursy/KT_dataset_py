from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
dfCotaParlamentar = pd.read_csv('../input/cota_parlamentar_sp.csv', delimiter=',')
dfCotaParlamentar.dataframeName = 'cota_parlamentar_sp.csv'
nRow, nCol = dfCotaParlamentar.shape
print(f'{nRow} linhas por {nCol} colunas')
dfCotaParlamentar.info()
dfCotaParlamentar.head(15)
dfCotaParlamentar.drop(axis=1, columns={'datemissao','nudeputadoid','txtdescricaoespecificacao'}, inplace=True)
dfCotaParlamentar
dfCotaParlamentar.vlrdocumento = dfCotaParlamentar.vlrdocumento / 100
dfCotaParlamentar
dfCotaParlamentar.rename(index=str, columns={
'nulegislatura' : 'Inicio_Legislatura',
'numano' : 'Ano',
'nummes' : 'Mes',
'sgpartido' : 'Partido',
'txnomeparlamentar' : 'Nome_Parlamentar','txtdescricao' : 'Categoria',
'txtfornecedor' : 'Nome_Fornecedor',
'vlrdocumento' : 'Valor'
}, inplace=True)
dfCotaParlamentar
dfTotalGastos = dfCotaParlamentar[['Ano','Valor']].groupby(by='Ano').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfTotalGastos.index = dfTotalGastos.index.map(int)
dfTotalGastos
dfAno = dfCotaParlamentar.Ano.value_counts().to_frame('TotalComprovantes').sort_index()
dfAno
dfGastosComprovantes = dfAno.join(dfTotalGastos)
dfGastosComprovantes
trace = go.Scatter(
                x = dfAno.index,
                y = dfAno.TotalComprovantes,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfAno.index)
layout = go.Layout(
    title='Notas fiscais apresentadas por ano',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de notas fiscais',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
trace = go.Scatter(
                x = dfTotalGastos.index,
                y = dfTotalGastos.TotalGastos / 1000000,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfTotalGastos.index)
layout = go.Layout(
    title='Total gasto pelos parlamentares por ano (em milhões de R$)',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total gasto (milhões de R$)',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
trace = go.Scatter(
                x = dfGastosComprovantes.index,
                y = dfGastosComprovantes.TotalComprovantes,
                mode = 'markers',
                marker=dict(
                    size= dfGastosComprovantes.TotalGastos / 100000,
                    color = dfGastosComprovantes.TotalGastos,
                    colorscale='Jet',
                    showscale=True
                ),
                text = dfGastosComprovantes.TotalGastos / 1000000)

layout = go.Layout(
    title='Correlação entre gastos anuais (em milhões de R$) e quantidade de comprovantes apresentados',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total de comprovantes apresentados',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfParlamentaresPorPartido = dfCotaParlamentar[['Partido','Nome_Parlamentar']].drop_duplicates().groupby(by="Partido").count().rename(index=str, columns={'Nome_Parlamentar':'Qtde_Parlamentares'})

trace = go.Pie(
                labels = dfParlamentaresPorPartido.index,
                values = dfParlamentaresPorPartido.Qtde_Parlamentares,
                text = dfParlamentaresPorPartido.index)
layout = go.Layout(
    title='Quantidade de parlamentares por partido/legenda (de 2009 à 2018)'
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfPartidos = dfCotaParlamentar.Partido.value_counts().to_frame('CotaPorPartido')
dfPartidos

trace = go.Pie(
                labels = dfPartidos.index,
                values = dfPartidos.CotaPorPartido,
                text = dfPartidos.index)
layout = go.Layout(
    title='Quantidade de gastos reportados por partido/legenda (de 2009 à 2018)'
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfGastosPorPartido = dfCotaParlamentar[['Partido','Valor']].groupby(by="Partido").sum().rename(index=str, columns={'Valor':'Total_Gastos'})

trace = go.Pie(
                labels = dfGastosPorPartido.index,
                values = dfGastosPorPartido.Total_Gastos / 1000000,
                text = dfGastosPorPartido.index)
layout = go.Layout(
    title='Total de gastos efetuados (em milhões de R$) por partido/legenda (de 2009 à 2018)'
)

py.iplot(go.Figure(data = [trace], layout=layout))

dfParlamentares = dfCotaParlamentar.Nome_Parlamentar.value_counts().to_frame('CotaPorParlamentar').sort_index()
dfParlamentares

trace = go.Bar(
                x = dfParlamentares.index,
                y = dfParlamentares.CotaPorParlamentar,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfParlamentares.index)
layout = go.Layout(
    title='Quantidade de gastos reportados por parlamentar (de 2009 à 2018)',
    xaxis=dict(
        title='Nome do Parlamentar',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de gastos reportados',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfParlamentares = dfCotaParlamentar[['Nome_Parlamentar','Valor']].groupby(by="Nome_Parlamentar").sum().rename(index=str, columns={'Valor':'Total_Gastos'}).sort_index()
dfParlamentares

trace = go.Bar(
                x = dfParlamentares.index,
                y = dfParlamentares.Total_Gastos / 1000000,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfParlamentares.index)
layout = go.Layout(
    title='Total de gastos efetuados (em milhões de R$) por parlamentar (de 2009 à 2018)',
    xaxis=dict(
        title='Nome do Parlamentar',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Valor total gasto',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfNotasPorFornecedor = dfCotaParlamentar.Nome_Fornecedor.value_counts().to_frame('NotasPorFornecedor').sort_index().sort_values(by='NotasPorFornecedor', ascending=False).head(30)
dfNotasPorFornecedor

trace = go.Bar(
                x = dfNotasPorFornecedor.index,
                y = dfNotasPorFornecedor.NotasPorFornecedor,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfNotasPorFornecedor.index)
layout = go.Layout(
    title='Total de gastos reportados por fornecedor (de 2009 à 2018 - Apenas os 25 com maior gasto)',
    xaxis=dict(
        title='Nome do Fornecedor',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de gastos reportados',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfGastosPorFornecedor = dfCotaParlamentar[['Nome_Fornecedor','Valor']].groupby(by="Nome_Fornecedor").sum().rename(index=str, columns={'Valor':'Total_Gastos'}).sort_values(by='Total_Gastos', ascending=False).head(30)

trace = go.Bar(
                x = dfGastosPorFornecedor.index,
                y = dfGastosPorFornecedor.Total_Gastos / 1000000,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfGastosPorFornecedor.index)
layout = go.Layout(
    title='Total de gastos efetuados (em milhões de R$) por fornecedor (de 2009 à 2018 - Apenas os 25 com maior gasto)',
    xaxis=dict(
        title='Nome do Fornecedor',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Valor total gasto',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfNotasPorCategoria = dfCotaParlamentar.Categoria.value_counts().to_frame('NotasPorCategoria').sort_index()
dfNotasPorCategoria

trace = go.Bar(
                x = dfNotasPorCategoria.index,
                y = dfNotasPorCategoria.NotasPorCategoria,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfNotasPorCategoria.index)
layout = go.Layout(
    title='Total de gastos reportados por categoria (de 2009 à 2018)',
    xaxis=dict(
        title='Categoria',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de gastos reportados',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfGastosPorCategoria = dfCotaParlamentar[['Categoria','Valor']].groupby(by="Categoria").sum().rename(index=str, columns={'Valor':'Total_Gastos'}).sort_index()

trace = go.Bar(
                x = dfGastosPorCategoria.index,
                y = dfGastosPorCategoria.Total_Gastos / 1000000,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfGastosPorCategoria.index)
layout = go.Layout(
    title='Total de gastos efetuados (em milhões de R$) por categoria (de 2009 à 2018)',
    xaxis=dict(
        title='Categoria',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Valor total gasto',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfGastosPorMes = dfCotaParlamentar[['Mes','Valor']].groupby(by='Mes').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfGastosPorMes.index = dfGastosPorMes.index.map(int)
dfGastosPorMes
dfNotasPorMes = dfCotaParlamentar.Mes.value_counts().to_frame('TotalNotas').sort_index()
dfNotasPorMes
dfNotasGastosMensais = dfNotasPorMes.join(dfGastosPorMes)
dfNotasGastosMensais
trace = go.Scatter(
                x = dfNotasPorMes.index,
                y = dfNotasPorMes.TotalNotas,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfNotasPorMes.index)
layout = go.Layout(
    title='Notas fiscais apresentadas por mês',
    xaxis=dict(
        title='Mês',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de notas apresentadas',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data=[trace], layout=layout))
trace = go.Scatter(
                x = dfGastosPorMes.index,
                y = dfGastosPorMes.TotalGastos / 1000000,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfGastosPorMes.index)
layout = go.Layout(
    title='Total gasto pelos parlamentares por mês (em milhões de R$)',
    xaxis=dict(
        title='Mês',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total gasto (milhões de R$)',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
trace = go.Scatter(
                x = dfNotasGastosMensais.index,
                y = dfNotasGastosMensais.TotalNotas,
                mode = 'markers',
                marker=dict(
                    size= dfNotasGastosMensais.TotalGastos / 100000,
                    color = dfNotasGastosMensais.TotalGastos,
                    colorscale='Jet',
                    showscale=True
                ),
                text = dfNotasGastosMensais.TotalGastos / 1000000)

layout = go.Layout(
    title='Correlação entre gastos anuais (em milhões de R$) e quantidade de comprovantes apresentados',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total de comprovantes apresentados',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
x = dfCotaParlamentar.Valor
data = [go.Histogram(x=x, nbinsx = 50)]

layout = go.Layout(
    title='Histograma - Valores por nota fiscal',
    xaxis=dict(
        title='Valor da nota'
    ),
    yaxis=dict(
        title='Quantidade'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
mask = (dfCotaParlamentar.Valor >= -250) & (dfCotaParlamentar.Valor <= 250)
x = dfCotaParlamentar.Valor[mask]
data = [go.Histogram(x=x, nbinsx = 50)]

layout = go.Layout(
    title='Valores por nota fiscal - Reembolsos até R$ 250,00 a gastos de R$ 250,00',
    xaxis=dict(
        title='Valor da nota/reembolso'
    ),
    yaxis=dict(
        title='Quantidade'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
mask = (dfCotaParlamentar.Valor >= -5) & (dfCotaParlamentar.Valor <= 4.99)
x = dfCotaParlamentar.Valor[mask]
data = [go.Histogram(x=x, nbinsx = 50)]

layout = go.Layout(
    title='Valores por nota fiscal - Reembolsos até R$ 5,00 à gastos de R$ 4,99',
    xaxis=dict(
        title='Valor da nota/reembolso'
    ),
    yaxis=dict(
        title='Quantidade'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
mask = (dfCotaParlamentar.Valor >= 0) & (dfCotaParlamentar.Valor <= 1.59)
x = dfCotaParlamentar.Valor[mask]
data = [go.Histogram(x=x, nbinsx = 50)]

layout = go.Layout(
    title='Valores por nota fiscal - Gastos até R$ 1,59',
    xaxis=dict(
        title='Valor da nota/reembolso'
    ),
    yaxis=dict(
        title='Quantidade'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
