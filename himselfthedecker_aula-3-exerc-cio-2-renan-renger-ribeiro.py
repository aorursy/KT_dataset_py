import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
dfCotaParlamentar = pd.read_csv('../input/deputies-parlamentary-quota-sp-20092018/cota_parlamentar_sp.csv', delimiter=',')
dfCotaParlamentar.dataframeName = 'cota_parlamentar_sp.csv'
dfCotaParlamentar.drop(axis=1, columns={'datemissao','nudeputadoid','txtdescricaoespecificacao'}, inplace=True)
dfCotaParlamentar.vlrdocumento = dfCotaParlamentar.vlrdocumento / 100
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
dfGastosPorMes = dfCotaParlamentar[['Mes','Valor']].groupby(by='Mes').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfGastosPorMes.index = dfGastosPorMes.index.map(int)
dfNotasPorMes = dfCotaParlamentar.Mes.value_counts().to_frame('TotalNotas').sort_index()
dfNotasGastosMensais = dfNotasPorMes.join(dfGastosPorMes)
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
    title='Correlação entre gastos mensais (em milhões de R$) e quantidade de comprovantes apresentados',
    xaxis=dict(
        title='Mês',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total de notas fiscais entregues',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfNotasGastosMensais['ValorMedioPorNota'] = dfNotasGastosMensais.TotalGastos / dfNotasGastosMensais.TotalNotas
dfNotasGastosMensais['IndiceGastosMensais'] = (dfNotasGastosMensais.TotalNotas / dfNotasGastosMensais.TotalGastos) * 100
dfNotasGastosMensais['MediaAnualNotas'] = dfNotasGastosMensais.ValorMedioPorNota.mean()
dfNotasGastosMensais['MediaAnualIndice'] = dfNotasGastosMensais.IndiceGastosMensais.mean()
trace = go.Scatter(
                x = dfNotasGastosMensais.index,
                y = dfNotasGastosMensais.ValorMedioPorNota,
                name = 'Mensal',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)))

traceAvg = go.Scatter(
                x = dfNotasGastosMensais.index,
                y = dfNotasGastosMensais.MediaAnualNotas,
                name = 'Média anual',
                mode = 'lines',
                marker = dict(color = 'rgba(168, 69, 227, 0.9)', line=dict(color='rgb(0,0,0)',width=0)))

layout = go.Layout(
    title='Valor médio das notas apresentadas (em R$) por mês',
    xaxis=dict(
        title='Mês',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Valor médio das notas apresentadas',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace,traceAvg], layout=layout))
trace = go.Scatter(
                x = dfNotasGastosMensais.index,
                y = dfNotasGastosMensais.IndiceGastosMensais,
                name = 'Mensal',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)))

traceAvg = go.Scatter(
                x = dfNotasGastosMensais.index,
                y = dfNotasGastosMensais.MediaAnualIndice,
                name = 'Média anual',
                mode = 'lines',
                marker = dict(color = 'rgba(168, 69, 227, 0.9)', line=dict(color='rgb(0,0,0)',width=0)))

layout = go.Layout(
    title='Índice de notas fiscais apresentadas por mês',
    xaxis=dict(
        title='Mês',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Índice (escala 0 - 100)',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace,traceAvg], layout=layout))
menorValor = dfNotasGastosMensais.ValorMedioPorNota.min()
menorIndice = dfNotasGastosMensais.IndiceGastosMensais.max()
diffValor = (dfNotasGastosMensais.loc[[12]].ValorMedioPorNota - menorValor).get_values()[0]
diffIndice = (dfNotasGastosMensais.loc[[12]].IndiceGastosMensais - menorIndice).get_values()[0]

print(f'No que tange a valor, a diferença é de R$ {round(diffValor,2)}, que corresponde a {round((diffValor / menorValor) * 100,2)}%')
print(f'Já no que tange ao índice, a diferença é de {round(diffIndice,2)}, que corresponde a {round((diffIndice / menorIndice) * 100,2)}%')
dfDezembro = dfCotaParlamentar[dfCotaParlamentar.Mes == 12]
dfDezembroPorAno = dfDezembro[['Ano','Valor']].groupby(by='Ano').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfDezembroPorAno['Media'] = dfDezembroPorAno.TotalGastos.mean()

trace = go.Bar(
                x = dfDezembroPorAno.index,
                y = round(dfDezembroPorAno.TotalGastos,2),
                name = 'Gasto mensal',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

traceAvg = go.Bar(
                x = dfDezembroPorAno.index,
                y = round(dfDezembroPorAno.Media,2),
                name = 'Média',
                marker = dict(color = 'rgba(168, 69, 227, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(
    title='Gastos (em R$) durante o mês de Dezembro, por ano',
    xaxis=dict(
        title='Ano',
        
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total gasto (em R$)',
        tickprefix = 'R$ ',
        hoverformat = ',2f',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace,traceAvg], layout=layout))
dfDezembroPorAno['DiffPorcentagem'] = ((dfDezembroPorAno.TotalGastos / dfDezembroPorAno.Media) - 1) * 100
dfDezembroPorAno

colorGuide=np.array(['rgba(255, 255, 255, 0.9)']*dfDezembroPorAno.shape[0])
colorGuide[dfDezembroPorAno.DiffPorcentagem<0]='rgba(47, 173, 38, 0.9)'
colorGuide[dfDezembroPorAno.DiffPorcentagem>=0]='rgba(173, 38, 44, 0.9)'

trace = go.Bar(
                x = dfDezembroPorAno.index,
                y = round(dfDezembroPorAno.DiffPorcentagem,2),
                name = 'Gasto mensal',
                marker = dict(
                    color = colorGuide.tolist(), line=dict(color='rgb(0,0,0)',width=1.5)),
)
layout = go.Layout(
    title='Percentual de divergência em relação a média por ano',
    xaxis=dict(
        title='Ano',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Percentual em relação a média',
        ticksuffix = '%',
        hoverformat = ',2f',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfPeriodoAlvo = dfCotaParlamentar[(dfCotaParlamentar.Ano == 2014) & (dfCotaParlamentar.Mes == 12)]
print(f'Período escolhido: {dfPeriodoAlvo.Mes.unique()[0]} / {dfPeriodoAlvo.Ano.unique()[0]}')
print(f'Total de notas apresentadas: {dfPeriodoAlvo.Valor.count()}')
print('Total gasto em R$ {:0,.2f}'.format(round(dfPeriodoAlvo.Valor.sum(),2)))
x = dfPeriodoAlvo.Valor
data = [go.Histogram(x=x, nbinsx = 100)]

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
allValues = dfPeriodoAlvo.Valor
data = [go.Box(
    y=allValues,
    name='dfPeriodoAlvo.Valor'
)]

layout = go.Layout(
    title='Boxplot - Valores por nota fiscal',
    yaxis=dict(
        title='Valor da Nota'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dfPosBoxplot = dfPeriodoAlvo[(dfPeriodoAlvo.Valor >= -537.93) & (dfPeriodoAlvo.Valor <= 925.13)]
print(f'Total de notas filtradas: {dfPosBoxplot.Valor.count()}')
print('Total gasto em R$ {:0,.2f}'.format(round(dfPosBoxplot.Valor.sum(),2)))
print(f'Percentual de dados no dataset filtrado: { round(((dfPosBoxplot.Valor.count() / dfPeriodoAlvo.Valor.count()) * 100),2) }%')
print(f'Percentual do valor total no dataset filtrado: { round(((dfPosBoxplot.Valor.sum() / dfPeriodoAlvo.Valor.sum()) * 100),2) }%')
data = [go.Violin(
    y=dfPosBoxplot.Valor,
    name='dfPosBoxplot.Valor',
    box = dict(
        visible= True
    ),
    line= dict(
        color = 'black'
    ),
    meanline= dict(
        visible = True
    ),
)]

layout = go.Layout(
    title='Violino pós filtro - Valores por nota fiscal',
    yaxis=dict(
        title='Valor da Nota'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


dfPeriodoCategorias = dfPeriodoAlvo[['Categoria','Valor']].groupby(by='Categoria').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
data = [go.Scatterpolar(
  r = round(dfPeriodoCategorias.nlargest(7, 'TotalGastos').TotalGastos,2),
  theta = dfPeriodoCategorias.nlargest(7, 'TotalGastos').index,
  fill = 'toself'
)]

layout = go.Layout(
  title = 'Distribuição de gastos por categoria (em R$ - 7 maiores apenas)',  
  polar = dict(
    radialaxis = dict(
      visible = True
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dfBilhetesPeriodo = dfPeriodoAlvo[dfPeriodoAlvo.Categoria == 'Emissão Bilhete Aéreo']
dfBilhetesParlamentar = dfBilhetesPeriodo[['Nome_Parlamentar','Valor']].groupby(by='Nome_Parlamentar').sum().rename(index=str, columns={'Valor' :'TotalGastos'})
dfQtdeBilhetesParlamentar = dfBilhetesPeriodo.Nome_Parlamentar.value_counts().to_frame('QuantidadeViagens')
dfMaioresViajantes = dfBilhetesParlamentar.nlargest(15, 'TotalGastos')
trace = go.Bar(
                x = dfMaioresViajantes.index,
                y = round(dfMaioresViajantes.TotalGastos,2),
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(
    title='Gastos (em R$) durante o mês de Dezembro com bilhetes aéreos',
    xaxis=dict(
        title='Parlamentar',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total gasto (em R$)',
        tickprefix = 'R$ ',
        hoverformat = ',2f',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfMaisFrequentes = dfQtdeBilhetesParlamentar.nlargest(15, 'QuantidadeViagens')
trace = go.Bar(
                x = dfMaisFrequentes.index,
                y = round(dfMaisFrequentes.QuantidadeViagens,2),
                name = 'Gasto mensal',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(
    title='Quantidade de bilhetes emitidos por parlamentar no mês de Dezembro',
    xaxis=dict(
        title='Parlamentar',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total de bilhetes emitidos',
        hoverformat = ',2f',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
dfQtdeIndividual = dfQtdeBilhetesParlamentar[
    (dfQtdeBilhetesParlamentar.index == 'MARA GABRILLI') |
    (dfQtdeBilhetesParlamentar.index == 'LUIZA ERUNDINA') |
    (dfQtdeBilhetesParlamentar.index == 'ARLINDO CHINAGLIA')
]
dfCustosIndividual =  dfBilhetesParlamentar[
    (dfBilhetesParlamentar.index == 'MARA GABRILLI') |
    (dfBilhetesParlamentar.index == 'LUIZA ERUNDINA') |
    (dfBilhetesParlamentar.index == 'ARLINDO CHINAGLIA')
]
traceQtde = go.Bar(
                x = dfQtdeIndividual.index,
                y = round(dfQtdeIndividual.QuantidadeViagens,2) * 100,
                name = 'Bilhetes emitidos (x 100)',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
traceCustos = go.Bar(
                x = dfCustosIndividual.index,
                y = round(dfCustosIndividual.TotalGastos,2),
                name = 'Custo total (em R$)',
                marker = dict(color = 'rgba(168, 69, 227, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(
    title='Quantidade de bilhetes emitidos (x 100) vs Custos com viagens',
    xaxis=dict(
        title='Parlamentar',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Total de bilhetes emitidos',
        hoverformat = ',2f',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [traceQtde, traceCustos], layout=layout))