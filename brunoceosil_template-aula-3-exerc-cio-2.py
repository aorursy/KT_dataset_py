import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)
#Importando dataset
url_linha = "../input/cota-parlamentar-sp/cota_parlamentar_sp.csv"
dslinha = pd.read_csv(url_linha, sep=',')

dslinha.head(1)
#Verificando os partidos que mais gastaram
by_part = dslinha.groupby(['sgpartido'])['vlrdocumento'].agg('sum').sort_values(ascending=False)
by_part.head(5)
#Construindo o gráfico de gastos por partidos por anos

#Pegando os 5 partidos que mais gastaram
by_year_PT   = dslinha[dslinha['sgpartido'] == 'PT'].groupby(['numano'])['vlrdocumento'].agg('sum')
by_year_PSDB = dslinha[dslinha['sgpartido'] == 'PSDB'].groupby(['numano'])['vlrdocumento'].agg('sum')
by_year_PP   = dslinha[dslinha['sgpartido'] == 'PP'].groupby(['numano'])['vlrdocumento'].agg('sum')
by_year_DEM  = dslinha[dslinha['sgpartido'] == 'DEM'].groupby(['numano'])['vlrdocumento'].agg('sum')
by_year_PR   = dslinha[dslinha['sgpartido'] == 'PR'].groupby(['numano'])['vlrdocumento'].agg('sum')

#Pegando os anos em que houveram gastos
year = dslinha['numano'].sort_values().unique()

#Criando as linhas de dados
linha1 = go.Scatter(x = year, y = by_year_PT, marker =  {'color' : 'red'}, mode = 'markers+lines', name = 'PT')
linha2 = go.Scatter(x = year, y = by_year_PSDB, marker =  {'color' : 'blue'}, mode = 'markers+lines', name = 'PSDB')
linha3 = go.Scatter(x = year, y = by_year_PP, marker =  {'color' : 'yellow'}, mode = 'markers+lines', name = 'PP')
linha4 = go.Scatter(x = year, y = by_year_DEM, marker =  {'color' : 'green'}, mode = 'markers+lines', name = 'DEM')
linha5 = go.Scatter(x = year, y = by_year_PR, marker =  {'color' : 'gray'}, mode = 'markers+lines', name = 'PR')

dados = [linha1, linha2, linha3, linha4, linha5]

#Criando o layout do gráfico

layout = go.Layout(title='5 Partidos com Maiores Gastos entre 2009 e 2018 - SP',
                   yaxis={'title':'Valor Gasto'},
                   xaxis={'title':'Ano do Gasto'})

fig = go.Figure(data=dados, layout=layout)

py.iplot(fig)
#Importando dataset
url_barra = "../input/aula3-ex1/BR_eleitorado_2016_municipio.csv"
dsbarra = pd.read_csv(url_barra, sep=',')
dsbarra.head(3)
#Trablhando o dataset para incluir as regioes

dsbarra['regiao'] = 'Nordeste'
dsbarra['regiao'][(dsbarra['uf'] == 'RS') | (dsbarra['uf'] == 'PR') | (dsbarra['uf'] == 'SC')] = 'Sul'
dsbarra['regiao'][(dsbarra['uf'] == 'SP') | (dsbarra['uf'] == 'RJ') | (dsbarra['uf'] == 'ES') | (dsbarra['uf'] == 'MG')] = 'Sudeste'
dsbarra['regiao'][(dsbarra['uf'] == 'GO') | (dsbarra['uf'] == 'DF') | (dsbarra['uf'] == 'MT') | (dsbarra['uf'] == 'MS')] = 'Centro-Oeste'
dsbarra['regiao'][(dsbarra['uf'] == 'AC') | (dsbarra['uf'] == 'AM') | (dsbarra['uf'] == 'RR') | (dsbarra['uf'] == 'PA') | (dsbarra['uf'] == 'TO') | (dsbarra['uf'] == 'RO') | (dsbarra['uf'] == 'AP')] = 'Norte'
dsbarra.head(10)
#Construindo o gráfico eleitores por região do Brasil

#Agrupando por região
by_region_tot   = dsbarra.groupby(['regiao'])['total_eleitores'].agg('sum')
by_region_man   = dsbarra.groupby(['regiao'])['gen_masculino'].agg('sum')
by_region_wom   = dsbarra.groupby(['regiao'])['gen_feminino'].agg('sum')


#Pegando as regiões
region = dsbarra['regiao'].sort_values().unique()

#Criando as barras de dados
barra1 = go.Bar(x = region, y = by_region_tot, marker =  {'color' : 'red'},name = 'Total')
barra2 = go.Bar(x = region, y = by_region_man, marker =  {'color' : 'blue'},name = 'Homens')
barra3 = go.Bar(x = region, y = by_region_wom, marker =  {'color' : 'yellow'}, name = 'Mulheres')

dados = [barra1, barra2, barra3]

#Criando o layout do gráfico

layout = go.Layout(title='Eleitores por Regiões do Brasil',
                   yaxis={'title':'Número de Eleitores'},
                   xaxis={'title':'Região do Brasil'})

fig = go.Figure(data=dados, layout=layout)

py.iplot(fig)
#Importando dataset
url_hist = "../input/cota-parlamentar-sp/cota_parlamentar_sp.csv"
dshist = pd.read_csv(url_hist, sep=',')
dshist.head(3)
#Construindo o histograma de distribuição quantidade de notas por parlamentar

#Criando os dados para o histograma
histo = go.Histogram(y = dshist['txnomeparlamentar'])

dados = [histo]

py.iplot(dados)
url_radar = "https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv"
dsradar = pd.read_csv(url_radar, sep=',')
dsradar[(dsradar['Name'] == 'Pikachu') | (dsradar['Name'] == 'Mewtwo')]
#Criando os atributos para plotar
hp   = int(dsradar[dsradar['Name'] == 'Pikachu']['HP'])
atq  = int(dsradar[dsradar['Name'] == 'Pikachu']['Attack'])
defs = int(dsradar[dsradar['Name'] == 'Pikachu']['Defense'])
satq = int(dsradar[dsradar['Name'] == 'Pikachu']['Sp. Atk'])
sdef = int(dsradar[dsradar['Name'] == 'Pikachu']['Sp. Def'])
spe  = int(dsradar[dsradar['Name'] == 'Pikachu']['Speed'])

hp1   = int(dsradar[dsradar['Name'] == 'Mewtwo']['HP'])
atq1  = int(dsradar[dsradar['Name'] == 'Mewtwo']['Attack'])
defs1 = int(dsradar[dsradar['Name'] == 'Mewtwo']['Defense'])
satq1 = int(dsradar[dsradar['Name'] == 'Mewtwo']['Sp. Atk'])
sdef1 = int(dsradar[dsradar['Name'] == 'Mewtwo']['Sp. Def'])
spe1  = int(dsradar[dsradar['Name'] == 'Mewtwo']['Speed'])

#Plotando a comparação
data = [
    go.Scatterpolar(
      r = [hp, atq, defs, satq, sdef, spe],
      theta = ['HP','ATTACK','DEFENSE', 'SP. ATK', 'SP. DEF', 'SPEED'],
      fill = 'toself',
      name = 'PIKACHU'
    ),
    go.Scatterpolar(
      r = [hp1, atq1, defs1, satq1, sdef1, spe1],
      theta = ['HP','ATTACK','DEFENSE', 'SP. ATK', 'SP. DEF', 'SPEED'],
      fill = 'toself',
      name = 'MEWTWO'
    )
]

layout = go.Layout(title = 'Reedição Filme Pokemon 2000 - Pikachu vs Mewtwo',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 160]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "radar/multiple")
#Construindo o gráfico eleitores por cidades nas região do Brasil

#Criando as caixas de dados
caixa1 = go.Box(y = dsbarra.loc[dsbarra['regiao'] == 'Centro-Oeste', 'total_eleitores'], marker =  {'color' : 'red'}, name = 'Centro-Oeste')
caixa2 = go.Box(y = dsbarra.loc[dsbarra['regiao'] == 'Nordeste', 'total_eleitores'], marker =  {'color' : 'blue'},name = 'Nordeste')
caixa3 = go.Box(y = dsbarra.loc[dsbarra['regiao'] == 'Norte', 'total_eleitores'], marker =  {'color' : 'yellow'}, name = 'Norte')
caixa4 = go.Box(y = dsbarra.loc[dsbarra['regiao'] == 'Sudeste', 'total_eleitores'], marker =  {'color' : 'green'}, name = 'Sudeste')
caixa5 = go.Box(y = dsbarra.loc[dsbarra['regiao'] == 'Sul', 'total_eleitores'], marker =  {'color' : 'grey'}, name = 'Sul')

dados = [caixa1, caixa2, caixa3, caixa4, caixa5]

#Criando o layout do gráfico

layout = go.Layout(title='Eleitores por Regiões do Brasil - Dispersão nas Cidades',
                   yaxis={'title':'Número de Eleitores'},
                   xaxis={'title':'Região do Brasil'})

fig = go.Figure(data=dados, layout=layout)

py.iplot(fig)
fig = {
    "data": [{
        "type": 'violin',
        "y": dsbarra['regiao'],
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
        "x0": 'Cidades Brasil'
    }],
    "layout" : {
        "title": "Quantidades de Cidades com Eleitores por Região",
        "yaxis": {
            "zeroline": False,
        }
    }
}

py.iplot(fig, filename = 'violin/basic', validate = False)
