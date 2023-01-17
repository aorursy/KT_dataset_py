import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
dfCotaParlamentar = pd.read_csv('../input/cota_parlamentar_sp.csv', delimiter=',')
dfCotaParlamentar.dataframeName = 'cota_parlamentar_sp.csv'
dfCotaParlamentar.head(15)

classificacao = [["datemissao", "Qualitativa Ordenal"],
                ["nudeputadoid","Qualitativa Nominal"],
                ["numlegislatura","Qualitativa Ordenal"],
                ["numano","Qualitativa Ordenal"],
                ["nummes","Qualitativa Ordenal"],
                ["sgpartido","Qualitativa Nominal"],
                ["txnomeparlamentar","Qualitativa Nominal"],
                ["txtdescricao","Qualitativa Nominal"],
                ["txtdescricaoespecificacao","Qualitativa Nominal"],
                ["txtfornecedor","Qualitativa Nominal"],
                ["vlrdocumento","Qualitativa Discreta"]]
classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])
classificacao
dfCotaParlamentar.vlrdocumento = dfCotaParlamentar.vlrdocumento / 100
dfCotaParlamentar
dfCotaParlamentar.rename(index=str, columns={'nulegislatura' : 'Inicio_Legislatura','numano' : 'Ano','nummes' : 'Mes','sgpartido' : 'Partido','txnomeparlamentar' : 'Nome_Parlamentar','txtdescricao' : 'Categoria','txtfornecedor' : 'Nome_Fornecedor','vlrdocumento' : 'Valor'}, inplace=True)
dfCotaParlamentar
dfTotalGastos = dfCotaParlamentar[['Ano','Valor']].groupby(by='Ano').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfTotalGastos.index = dfTotalGastos.index.map(int)
dfTotalGastos
dfTotalGastosPartido = dfCotaParlamentar[['Partido','Valor']].groupby(by='Partido').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfTotalGastosPartido.index = dfTotalGastosPartido.index.map(str)
dfTotalGastosPartido
dfTotalGastosPartidoPorAno = dfCotaParlamentar[['Ano','Partido','Valor']].groupby(by=['Ano','Partido']).sum().rename(index=str, columns={'Valor' :'TotalGastos'})

dfTotalGastosPartidoPorAno
dfTotalGastosPartidoCandidato = dfCotaParlamentar[['Partido','Nome_Parlamentar','Valor']].groupby(by=['Partido','Nome_Parlamentar']).sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfTotalGastosPartidoCandidato
dfTotalGastosCategoria = dfCotaParlamentar[['Categoria','Valor']].groupby(by=['Categoria']).sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfTotalGastosCategoria
trace = go.Scatter(
                x = dfTotalGastos.index,
                y = dfTotalGastos.TotalGastos,
                marker = dict(color = 'blue', line=dict(color='red',width=2.0)),
                text = dfTotalGastos.index)
layout = go.Layout(
    title='Total de Gastos por ano',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Gastos em Milhoes',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
trace = go.Bar(
                x = dfTotalGastosPartido.index,
                y = dfTotalGastosPartido.TotalGastos,
                marker = dict(color = 'purple', line=dict(color='black',width=1.5)),
                text = dfTotalGastos.index)
layout = go.Layout(
    title='Total de Gastos por Partido',
    xaxis=dict(
        title='Partido',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Gastos em Milhoes',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
trace = go.Bar(
                x = dfTotalGastosCategoria.index,
                y = dfTotalGastosCategoria.TotalGastos,
                marker = dict(color = 'darkorange', line=dict(color='black',width=2.0)),
                text = dfTotalGastosCategoria.index)
layout = go.Layout(
    title='Total de Gastos por Categoria',
    xaxis=dict(
        title='Categoria',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Gastos em Milhoes',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
