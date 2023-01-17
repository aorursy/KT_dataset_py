# Importar dados cota_parlamentar_sp.csv
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

py.init_notebook_mode(connected=True)

url = "../input/cota-parlamentar-sp/cota_parlamentar_sp.csv"
df = pd.read_csv(url, sep=',')
df.head(1)

# Filtrar dados
'''
importante notar que, para realizar uma análise de gastos anuais, é necessário manter uma análise comparativa mes a mes, 
portanto é preciso possuir dados de todos os meses do ano para possuir uma amostragem mais controlada. 
Para isto, foram desconsiderados os dados de 2009 e 2018.

df = df[df.numano > 2009]
df_novo = df[df.numano < 2018]
OU
'''
df_novo = df[(df['numano']  > 2009) & (df['numano'] < 2018)]
# Um gráfico de linha (Soma de vlrdocumento por Ano + linha de tendencia e previsao de 4 anos com intervalo de confianca de 90%
'''
será analisado a tendencia de gasto anual. É importante notar que houve um aumento de gasto. 200mi a cada 4 anos, 
portanto 16,6% de crescimento a cada 4 anos, isso pode ser superior, inclusive, ao valor da inflação acumulada. 
Portanto é importante possuir um controle de gastos parlamentares.
'''
#Arrumar dados 
# 1. Soma de vlrdocumento por Ano
vlr_total_ano   = df_novo.groupby(['numano'])['vlrdocumento'].agg('sum')
# 2. criando linha de tendência e e previsao de 4 anos com intervalo de confianca de 90%


# gerando um grafico de linha
dado1 = go.Scatter(x = vlr_total_ano.index, y = vlr_total_ano, marker =  {'color' : 'blue'}, mode = 'markers+lines')
dados = [dado1]
#Criando o layout do gráfico
layout = go.Layout(title='Gasto parlamentar por ano (2010 a 2017)',
                   yaxis={'title':'Valor (R$)'},
                   xaxis={'title':'Ano'})
fig = go.Figure(data=dados, layout=layout)
py.iplot(fig)
# Grafico de Barras
'''
Analisar os cinco partidos que mais gastaram dentro da epoca de 2010 a 2017. Verificar no mesmo gráfico os valores gastos por partido por ano. 
Neste gráfico é possível verificar uma estratégia anual de gastos por partido.
'''

#Filtrar os 5 partidos que mais gastaram entre 2010 e 2017 e criar dataframe ordenado dos gastos por partido por ano
top5_partidos = df_novo.groupby(['sgpartido'])['vlrdocumento'].agg('sum').sort_values(ascending=False).nlargest(5)

ano_2010 = df_novo[(df_novo['numano'] == vlr_total_ano.index[0])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2011 = df_novo[(df_novo['numano'] == vlr_total_ano.index[1])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2012 = df_novo[(df_novo['numano'] == vlr_total_ano.index[2])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2013 = df_novo[(df_novo['numano'] == vlr_total_ano.index[3])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2014 = df_novo[(df_novo['numano'] == vlr_total_ano.index[4])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2015 = df_novo[(df_novo['numano'] == vlr_total_ano.index[5])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2016 = df_novo[(df_novo['numano'] == vlr_total_ano.index[6])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)
ano_2017 = df_novo[(df_novo['numano'] == vlr_total_ano.index[7])&(df_novo.sgpartido.isin(top5_partidos.index))].groupby(['sgpartido'])['vlrdocumento'].agg('sum').reindex(top5_partidos.index)

#Criar as barras de dados para o grafico
barra1 = go.Bar(x = top5_partidos.index, y = ano_2010, marker =  {'color' : 'red'},name = '2010')
barra2 = go.Bar(x = top5_partidos.index, y = ano_2011, marker =  {'color' : 'blue'},name = '2011')
barra3 = go.Bar(x = top5_partidos.index, y = ano_2012, marker =  {'color' : 'yellow'}, name = '2012')
barra4 = go.Bar(x = top5_partidos.index, y = ano_2013, marker =  {'color' : 'green'}, name = '2013')
barra5 = go.Bar(x = top5_partidos.index, y = ano_2014, marker =  {'color' : 'orange'}, name = '2014')
barra6 = go.Bar(x = top5_partidos.index, y = ano_2015, marker =  {'color' : 'black'}, name = '2015')
barra7 = go.Bar(x = top5_partidos.index, y = ano_2016, marker =  {'color' : 'lime'}, name = '2016')
barra8 = go.Bar(x = top5_partidos.index, y = ano_2017, marker =  {'color' : 'pink'}, name = '2017')
dados = [barra1, barra2, barra3, barra4, barra5, barra6, barra7, barra8]

#Criar o layout do gráfico
layout = go.Layout(title='Gastos anuais por partidos (2010 a 2017)', yaxis={'title':'Valor (R$)'},xaxis={'title':'Partido'})

fig = go.Figure(data=dados, layout=layout)

py.iplot(fig)



# Gráfico de Radar
'''
Neste gráfico é possível verificar a semelhança de despesas entre partidos, podendo verificar assim que, 
por este gráfico, não há uma diferença grande entre o gasto médio por partido dentro das categorias que mais geram despesas
'''


#Filtrar as 5 categorias que mais geraram despesas entre 2010 e 2017 e criar dataframe ordenado dos gastos médios por partido por categorias
top5_categorias = df_novo.groupby(['txtdescricao'])['vlrdocumento'].agg('sum').sort_values(ascending=False).nlargest(5)

df_ano_partido1 = df_novo[(df_novo['sgpartido'] == top5_partidos.index[0])&(df_novo.txtdescricao.isin(top5_categorias.index))].groupby(['txtdescricao'])['vlrdocumento'].mean().reindex(top5_categorias.index)
df_ano_partido2 = df_novo[(df_novo['sgpartido'] == top5_partidos.index[1])&(df_novo.txtdescricao.isin(top5_categorias.index))].groupby(['txtdescricao'])['vlrdocumento'].mean().reindex(top5_categorias.index)
df_ano_partido3 = df_novo[(df_novo['sgpartido'] == top5_partidos.index[2])&(df_novo.txtdescricao.isin(top5_categorias.index))].groupby(['txtdescricao'])['vlrdocumento'].mean().reindex(top5_categorias.index)
df_ano_partido4 = df_novo[(df_novo['sgpartido'] == top5_partidos.index[3])&(df_novo.txtdescricao.isin(top5_categorias.index))].groupby(['txtdescricao'])['vlrdocumento'].mean().reindex(top5_categorias.index)
df_ano_partido5 = df_novo[(df_novo['sgpartido'] == top5_partidos.index[4])&(df_novo.txtdescricao.isin(top5_categorias.index))].groupby(['txtdescricao'])['vlrdocumento'].mean().reindex(top5_categorias.index)

#Configurar radares
radar1 = go.Scatterpolar(r = df_ano_partido1.values, theta = top5_categorias.index, fill = 'none', name = top5_partidos.index[0])
radar2 = go.Scatterpolar(r = df_ano_partido2.values, theta = top5_categorias.index, fill = 'none', name = top5_partidos.index[1])
radar3 = go.Scatterpolar(r = df_ano_partido3.values, theta = top5_categorias.index, fill = 'none', name = top5_partidos.index[2])
radar4 = go.Scatterpolar(r = df_ano_partido4.values, theta = top5_categorias.index, fill = 'none', name = top5_partidos.index[3])
radar5 = go.Scatterpolar(r = df_ano_partido5.values, theta = top5_categorias.index, fill = 'none', name = top5_partidos.index[4])

data = [radar1, radar2, radar3, radar4, radar5]

#Criar o layout do gráfico
layout = go.Layout(polar = dict(radialaxis = dict(visible = True)),showlegend = True)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "radar/multiple")



df_top5_categorias = df_novo[df_novo.txtdescricao.isin(top5_categorias.index)][['txtdescricao','vlrdocumento']]



caixa1 = go.Box(y = df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[0]]['vlrdocumento'].values, marker =  {'color' : 'red'}, name = top5_categorias.index[0])
caixa2 = go.Box(y = df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[1]]['vlrdocumento'].values, marker =  {'color' : 'blue'}, name = top5_categorias.index[1])
caixa3 = go.Box(y = df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[2]]['vlrdocumento'].values, marker =  {'color' : 'yellow'}, name = top5_categorias.index[2])
caixa4 = go.Box(y = df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[3]]['vlrdocumento'].values, marker =  {'color' : 'green'}, name = top5_categorias.index[3])
caixa5 = go.Box(y = df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[4]]['vlrdocumento'].values, marker =  {'color' : 'grey'}, name = top5_categorias.index[4])
dados = [caixa1, caixa2, caixa3, caixa4, caixa5]

layout = go.Layout(title='Eleitores por Regiões do Brasil - Dispersão nas Cidades',
                   yaxis={'title':'Número de Eleitores'},
                   xaxis={'title':'Região do Brasil'})

fig = go.Figure(data=dados, layout=layout)

py.iplot(fig)


fig = {
    "data": [{
        "type": 'violin',
        "y": df_top5_categorias[df_top5_categorias['txtdescricao'] == top5_categorias.index[0]]['vlrdocumento'].values,
        "box": {
            "visible": True
        },
        "line": {
            "color": 'black'
        },
        "meanline": {
            "visible": True
        },
        "fillcolor": '#8dd3c7',
        "opacity": 0.6,
        "x0": top5_categorias.index[0]
    }],
    "layout" : {
        "title": "",
        "yaxis": {
            "zeroline": False,
        }
    }
}

py.iplot(fig, filename = 'violin/basic', validate = False)
