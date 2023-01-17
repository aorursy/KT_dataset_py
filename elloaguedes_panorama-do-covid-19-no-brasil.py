#Última execução

import datetime

print(datetime.datetime.now())

today = datetime.datetime.now().strftime('%d/%m/%Y')
# imports

import numpy as np

import pandas as pd

import os

import numpy as np



# bokeh packages

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

from bokeh.models import GeoJSONDataSource

output_notebook()



# plotly packages

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *



import json

import geopandas as gpd

import plotly.graph_objects as go

import unidecode
data = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')

data.head()
data.tail()
print(min(data['date']))

print(max(data['date']))
## Síntese diária

df2 = data.groupby(['date'])['cases','deaths'].agg('sum')

df2.head()
### Atualizando com antiga versão do dataset

old = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19_old.csv')

old = old.groupby(['date'])['suspects'].agg('sum')



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    xaxis = dict(

        tickmode = 'array',

        tickvals = old.index,

        ticktext = old.index

    ),

    xaxis_title="Data",

    yaxis_title = "Quantidade"

)

suspeitos = old.loc[:'2020-03-21']

fig = px.bar(title='Casos suspeitos -- Descontinuado a partir de 21/03/2020', x=suspeitos.index, y=suspeitos)

fig['layout'].update(layout)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

)



fig = make_subplots(rows=2, cols=1,subplot_titles=('Casos Confirmados até ' + today, 'Óbitos até '+ today))

fig.append_trace(go.Bar(name='Confirmados', x=df2.index, y=df2['cases']), row=1, col=1)

fig.append_trace(go.Bar(name='Óbitos', x=df2.index, y=df2['deaths']), row=2, col=1)



fig.update_xaxes(title_text="Data", row=1, col=1)

fig.update_yaxes(title_text="Quantidade", row=1, col=1)

fig.update_xaxes(title_text="Data", row=2, col=1)

fig.update_yaxes(title_text="Quantidade", row=2, col=1)



fig['layout'].update(layout)



fig.show()
import plotly.graph_objects as go



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Visualização Conjunta de Casos e Óbitos até " + today,

)



fig = go.Figure(data=[

    go.Bar(name='Confirmados', x=df2.index, y=df2['cases']),

    go.Bar(name='Óbitos', x=df2.index, y=df2['deaths'])

])

fig.update_xaxes(title_text='Data')

fig.update_yaxes(title_text='Quantidade')

fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
# utils

def remove_accents(a):

    unaccented_string = unidecode.unidecode(a)

    return unaccented_string
#data.drop('hour',axis= 1, inplace=True)

atual = max(data['date'])

df3 = data.loc[data['date'] == max(data['date'])].groupby(['state'])['cases','deaths'].agg('sum')

df4 = pd.DataFrame({"name": df3.index, 'cases': df3['cases'], 'deaths':df3['deaths']})

df4.index = range(0,27)



brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')



df4['name'] = df4['name'].apply(remove_accents)

df4 = df4.sort_values('name')

brazil['name'] = brazil['name'].apply(remove_accents)

brazil = brazil.sort_values('name')



pop_states = brazil.merge(df4, left_on = 'name', right_on = 'name')

geosource = GeoJSONDataSource(geojson = pop_states.to_json())

merged_json = json.loads(pop_states.to_json())

json_data = json.dumps(merged_json)

geosource = GeoJSONDataSource(geojson = json_data)
from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar

from bokeh.palettes import brewer

from bokeh.palettes import magma,viridis,cividis

from bokeh.layouts import row



def myplot3(geosource,tema, complemento = '',jump = 1,high = 100):

    

    tipo = 'Óbitos'

    palette = magma(256)

    if tema.startswith('case'):

        tipo = 'Casos'

        palette = viridis(256)[:248]

    elif tema.startswith('letalidade'):

        tipo = 'Letalidade'

        palette = cividis(256)[:248]

    elif tema.startswith('leitospor100mil'):

        tipo = 'Leitos de UTI por 100 mil habitantes'

        palette = magma(256)

    elif tema.startswith('leitos'):

        tipo = 'Leitos de UTI'

        palette = viridis(256)[:248]

    elif tema.startswith('testesRapidos'):

        tipo = 'Testes Rápidos'

        palette = viridis(256)[:248]

    elif tema.startswith('testesRTPCR'):

        tipo = 'Testes RT-PCR'

        palette = magma(256)

        

        

    palette = palette[::-1]

    color_mapper = LinearColorMapper(palette = palette, low = 0, high = high)



    #Define custom tick labels for color bar.

    if (not tema.startswith('letalidade')):

        d = {}

        for i in range(0,int(high),jump):

            d[str(i)] = str(i)



            

        d[str(int(high) + 1)] = '>' + str(int(high) + 1)

                

        hover = HoverTool(tooltips = [ ('Estado','@name'),('Quantidade', '@{'+tema+'}{%d}')], formatters={'@{'+ tema +'}' : 'printf'})

    elif (tema.startswith('leitos') or tema.startswith('teste')):

        d = {}

        for i in np.arange(0, high+1, jump):

            d[str(round(i,2))] = str(round(i,2))

        d[str(high + 1)] = '>'+ str(high + 1)

        hover = HoverTool(tooltips = [ ('Estado','@name'),('Quantidade', '@{'+tema+'}{%d}')], formatters={'@{'+ tema +'}' : 'printf'})

    else:

        d = {}

        for i in np.arange(0, high+0.5, jump):

            d[str(round(i,2))] = str(round(i,2))

        d[str(round(high + 0.5,2))] = '>'+ str(round(high + 0.5,2))

        hover = HoverTool(tooltips = [ ('Estado','@name'),('Taxa', '@{'+tema+'}{%.2f%%}')], formatters={'@{'+ tema +'}' : 'printf'})

    

    

    tick_labels = d

    #Create color bar. 

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 300, height = 20,

    border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)







    #Create figure object.

    p = figure(title = tipo + complemento + ' em {0}'.format((datetime.datetime.now()).strftime('%d/%m/%Y')), plot_height = 430 , plot_width = 330, toolbar_location = None, tools =[hover])

    p.xgrid.grid_line_color = None

    p.ygrid.grid_line_color = None

    p.xaxis.visible = False

    p.yaxis.visible = False





    p.patches('xs','ys', source = geosource,fill_color = {'field' :str(tema), 'transform' : color_mapper},

              line_color = 'black', line_width = 0.25, fill_alpha = 1)



    p.add_layout(color_bar, 'below')

    return p

show(row(myplot3(geosource = geosource,tema = 'cases',jump = 2000, high = max(df4['cases'])),

         myplot3(geosource = geosource,tema = 'deaths', jump = 1000, high = max(df4['deaths']))))
populacao = pd.read_csv('/kaggle/input/dadosbrasil/populacao.csv',sep=";")

populacao['name'] = populacao['name'].apply(remove_accents)

populacao = populacao.sort_values('name')

populacao = populacao.merge(df4, left_on = 'name', right_on = 'name')

populacao['casespor100mil'] = (populacao['cases']/populacao['populacao'])*100000

populacao['deathspor100mil'] = (populacao['deaths']/populacao['populacao'])*100000

populacao['leitospor100mil'] = (populacao['leitos']/populacao['populacao'])*100000

populacao['letalidade'] = round(populacao['deaths']/populacao['cases'],3)*100



# Abertura do mapa

brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')

## mesclagem das bases

brazil['name'] = brazil['name'].apply(remove_accents)

brazil = brazil.sort_values('name')

pop_states = brazil.merge(populacao, left_on = 'name', right_on = 'name')

# Input GeoJSON source that contains features for plotting

geosource = GeoJSONDataSource(geojson = pop_states.to_json())



import json



#Read data to json.

merged_json = json.loads(pop_states.to_json())

json_data = json.dumps(merged_json)
show(row(myplot3(geosource = geosource,tema = 'casespor100mil',jump = 10, high = max(populacao['casespor100mil']), complemento = ' por 100 mil habitantes '),

         myplot3(geosource = geosource,tema = 'deathspor100mil', jump = 2, high = max(populacao['deathspor100mil']), complemento = ' por 100 mil habitantes ')))
letalidade = sum(df4['deaths'])/sum(df4['cases'])

print("Taxa de Letalidade em " + today + ": {0:6.3f}%".format(letalidade*100))
df2['letalidade'] = df2['deaths']/df2['cases']

df2.fillna(0,inplace=True)

df2 = df2.reset_index()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Letalidade ao Longo do Tempo",

    xaxis_title="Data",

    yaxis_title="Taxa de Letalidade",

    yaxis_tickformat = '.2%')



fig = go.Figure(data=[

    go.Scatter(x=df2['date'], y=df2['letalidade'])])

fig['layout'].update(layout)



fig.show()
show(myplot3(geosource = geosource,tema = 'letalidade', jump = 0.5, high = max(populacao['letalidade']), complemento = ''))
show(row(myplot3(geosource = geosource,tema = 'leitos', jump = 1, high = max(populacao['leitos']), complemento = ''),myplot3(geosource = geosource,tema = 'leitospor100mil', jump = 1, high = max(populacao['leitospor100mil']), complemento = '')))
populacao['pib'] = [int(x.replace('.','')) for x in populacao['pib']]
populacao['pib'].corr(populacao['leitos'])
populacao['populacao'].corr(populacao['leitos'])
newData = pd.read_csv('/kaggle/input/testdata/testData.csv')

newData['name'] = newData['name'].apply(remove_accents)

newData = newData.merge(populacao, left_on = 'name', right_on = 'name')

newData['pibpercapita'].corr(newData['leitos'])
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "PIB per capita versus Leitos de UTI",

    xaxis_title="PIB per capita",

    yaxis_title="Leitos de UTI")



fig = go.Figure(data=[

    go.Scatter(x=newData['pibpercapita'], y=newData['leitos'],mode='markers')])

fig['layout'].update(layout)



fig.show()
populacao['idhm'] = [float(x.replace(',','.')) for x in populacao['idhm']]

populacao['idhm'].corr(populacao['leitos'])
newData = pd.read_csv('/kaggle/input/testdata/testData.csv')

newData['name'] = newData['name'].apply(remove_accents)

newData = newData.merge(populacao, left_on = 'name', right_on = 'name')

newData['pibpercapita'].corr(newData['letalidade'])
newData['letalidade'].corr(newData['leitos'])
newData = pd.read_csv('/kaggle/input/testdata/testData.csv')

newData['name'] = newData['name'].apply(remove_accents)

newData = newData.merge(df4, left_on = 'name', right_on = 'name')

newData['cases'].corr(newData['pibpercapita'])
## Cases on March 31/2020 per state

data = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')

df3 = data.loc[data['date'] == '2020-03-31'].groupby(['state'])['cases','deaths'].agg('sum')

df3 = df3.reset_index()

df3['name'] = df3['state'].apply(remove_accents)

df3.drop(['state'],axis=1,inplace=True)

newData = pd.read_csv('/kaggle/input/testdata/testData.csv')

newData['name'] = newData['name'].apply(remove_accents)

newData = newData.merge(df3, left_on = 'name', right_on = 'name')

newData['cases'].corr(newData['testesRapidos'])
newData['cases'].corr(newData['testesRTPCR'])
newData['deaths'].corr(newData['testesRapidos'])
newData['deaths'].corr(newData['testesRTPCR'])
subset = populacao[['name','pib']]

newData = newData.merge(subset, left_on = 'name', right_on = 'name')

newData['testesRapidos'].corr(newData['pib'])
newData['testesRTPCR'].corr(newData['pib'])
newData['pib'].corr(newData['cases'])
newData['pib'].corr(newData['deaths'])
populacao = pd.read_csv('/kaggle/input/dadosbrasil/populacao.csv',sep=";")

populacao['name'] = populacao['name'].apply(remove_accents)

populacao = populacao.sort_values('name')

newData = newData.merge(populacao,left_on='name', right_on='name')
newData['casescapita'] = (newData['cases']/newData['populacao'])

newData['deathscapita'] = (newData['deaths']/newData['populacao'])
newData['testesRapidos'].corr(newData['casescapita'])
newData['testesRapidos'].corr(newData['deathscapita'])
newData['testesRTPCR'].corr(newData['casescapita'])
newData['testesRTPCR'].corr(newData['deathscapita'])
# Abertura do mapa

brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')

brazil['name'] = brazil['name'].apply(remove_accents)

brazil = brazil.sort_values('name')

## mesclagem das bases

pop_states = brazil.merge(newData, left_on = 'name', right_on = 'name')

# Input GeoJSON source that contains features for plotting

geosource = GeoJSONDataSource(geojson = pop_states.to_json())
print("Dados de 31/03/2020, ignorar cabeçalho -- Data from 03/31/2020, ignore header")

show(row(myplot3(geosource = geosource,tema = 'testesRapidos', jump = 100, high = max(newData['testesRapidos']), complemento = ''),myplot3(geosource = geosource,tema = 'testesRTPCR', jump = 100, high = max(newData['testesRTPCR']), complemento = '')))
import plotly.graph_objects as go

df2 = data.groupby(['date'])['cases','deaths'].agg('sum')

df2 = df2.reset_index()



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Série temporal de Casos",

    xaxis_title="Data",

    yaxis_title="Quantidade",

)



fig = go.Figure(data=[

    go.Scatter(x=df2['date'], y=df2['cases'])

    

])

fig['layout'].update(layout)



fig.show()
dfy = df2.copy()

dfy.drop(['date','deaths'],axis=1,inplace = True)

dfy = dfy.reset_index()

dfy['dias'] = dfy['index']



# Cases double by rate every 2 days

def casesDouble(rate, doubleDays):

    supposedCases = [1]

    for i in range(len(doubleDays)-1):

        supposedCases.append(rate*supposedCases[-1])

    return supposedCases



doubleDays = list(range(0,max(dfy['dias']),2))

dfSupposed = pd.DataFrame({'dias':doubleDays,'2x':casesDouble(2,doubleDays),'3x':casesDouble(3,doubleDays),'1.5x':casesDouble(1.5,doubleDays)})
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Suposição do crescimento de casos a cada 2 dias (Escala logarítmica)",

    xaxis_title="Dias desde o primeiro caso",

    yaxis_title="Quantidade (escala log)",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3

    ),

    yaxis_type="log"

)



fig = go.Figure(data=[

    go.Scatter(x=dfy['dias'], y=dfy['cases'], name='Casos Reais',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['2x'], name = '2x',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['3x'], name = '3x',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['1.5x'], name = '1.5x',mode="lines+markers")

])



fig['layout'].update(layout)



fig.show()
dfy = df2.copy()

dfy.drop(['date','deaths'],axis=1,inplace = True)

dfy = dfy.reset_index()

dfy['dias'] = dfy['index']



# Cases double by rate every 3 days

def casesDouble(rate, doubleDays):

    supposedCases = [1]

    for i in range(len(doubleDays)-1):

        supposedCases.append(rate*supposedCases[-1])

    return supposedCases



doubleDays = list(range(0,max(dfy['dias']),3))

dfSupposed = pd.DataFrame({'dias':doubleDays,'2x':casesDouble(2,doubleDays),'3x':casesDouble(3,doubleDays),'1.5x':casesDouble(1.5,doubleDays)})
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Suposição do crescimento de casos a cada 3 dias (Escala logarítmica)",

    xaxis_title="Dias desde o primeiro caso",

    yaxis_title="Quantidade (escala log)",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3

    ),

    yaxis_type="log"

)



fig = go.Figure(data=[

    go.Scatter(x=dfy['dias'], y=dfy['cases'], name='Casos Reais',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['2x'], name = '2x',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['3x'], name = '3x',mode="lines+markers"),

    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['1.5x'], name = '1.5x',mode="lines+markers")

])



fig['layout'].update(layout)



fig.show()
import plotly.graph_objects as go

import datetime

import numpy as np



df2['date'] = pd.to_datetime(df2['date'])

df2 = df2.loc[df2['date'] >= '02-26-2020']

df2['dias'] = range(1,len(df2) + 1,1)



## Treino

dias_train = df2['dias'][:int(0.9*len(df2))]

cases_train = df2['cases'][:int(0.9*len(df2))]



## Teste

dias_test = df2['dias'][int(0.9*len(df2)):]

cases_test =  df2['cases'][int(0.9*len(df2)):]



previsao = len(df2) - len(dias_test)



print("Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dias" % (len(df2),len(dias_train),len(dias_test)))
from sklearn.linear_model import LinearRegression



reg = LinearRegression().fit(dias_train.values.reshape(-1,1), cases_train)

y_previsto = reg.predict(dias_test.values.reshape(-1,1))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Estimador linear para o número de casos",

    xaxis_title="Dias desde a primeira notificação",

    yaxis_title="Quantidade de casos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3

    )

)



fig = go.Figure(data=[

    go.Scatter(x=dias_train, y=df2['cases'][:int(0.9*len(df2))], name='Dados de Treinamento',mode="lines+markers"),

    go.Scatter(x=dias_test, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),

    go.Scatter(x=dias_test, y=df2['cases'][int(0.9*len(df2)):], name = 'Casos Reais',mode="lines+markers")

])

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0= previsao + 0.5,

            y0=120,

            x1=previsao + 0.5,

            y1=max(df2['cases']),

            line=dict(

                width=1.5,

                dash= "dash"

            )

))



fig.add_trace(go.Scatter(

    x=[previsao + 0.5],

    y=[2],

    text=["Início da previsão"],

    mode="text",

))

fig['layout'].update(layout)



fig.show()
from sklearn.metrics import mean_squared_error, r2_score

print("Erro médio quadrático: ",mean_squared_error(cases_test,y_previsto))

print("R^2 Score: ", r2_score(cases_test,y_previsto))
import plotly.graph_objects as go

import datetime

import numpy as np



df2['date'] = pd.to_datetime(df2['date'])

df2 = df2.loc[df2['date'] >= '02-26-2020']

df2['dias'] = range(1,len(df2) + 1,1)

log_y_data = np.log(df2['cases'])



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Log Casos versus Dias"

)



fig = go.Figure(data=[go.Scatter(name='log Casos',x=df2['dias'], y=log_y_data, mode='markers'),

                     go.Scatter(name='Referência',x=df2['dias'], y=df2['dias'], line=dict(color='firebrick', width=0.5,

                              dash='dash'))])

fig['layout'].update(layout)



fig.show()
import plotly.graph_objects as go

import datetime

import numpy as np



log_y_data = np.log(df2['cases'])



cases_train_log = log_y_data[:int(0.9*len(df2))]

cases_test_log = log_y_data[int(0.9*len(df2)):]



print("Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dias" % (len(df2),len(dias_train),len(dias_test)))
# Treino do modelo (interpolação da curva)

curve_fit = np.polyfit(dias_train, cases_train_log, 1)

y_train = (np.exp(curve_fit[1]) * np.exp(curve_fit[0]*dias_train)).astype(int)

y_estimado = (np.exp(curve_fit[1]) * np.exp(curve_fit[0]*dias_test)).astype(int)
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Estimador exponencial para o número de casos",

    xaxis_title="Dias desde a primeira notificação",

    yaxis_title="Quantidade de casos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3

    )

)



fig = go.Figure(data=[

    go.Scatter(x=dias_train, y=df2['cases'][:int(0.9*len(df2))], name='Dados de Treinamento',mode="lines+markers"),

    go.Scatter(x=dias_test, y=y_estimado, name = 'Casos Estimados',mode="lines+markers"),

    go.Scatter(x=dias_test, y=df2['cases'][int(0.9*len(df2)):], name = 'Casos Reais',mode="lines+markers")

])

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0= previsao + 0.5,

            y0=120,

            x1=previsao + 0.5,

            y1=max(y_estimado),

            line=dict(

                width=1.5,

                dash= "dash"

            )

))



fig.add_trace(go.Scatter(

    x=[previsao + 0.5],

    y=[2],

    text=["Início da previsão"],

    mode="text",

))

fig['layout'].update(layout)



fig.show()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("Erro médio quadrático: ",mean_squared_error(cases_test_log,y_estimado))

print("R^2 Score: ", r2_score(cases_test,y_estimado))
## Treino

dias_train = df2['dias'][:-1]

cases_train = df2['cases'][:-1]



## Teste

dias_test = df2['dias'][-1:]

cases_test =  df2['cases'][-1:]



previsao = len(df2) - len(dias_test)



print("Novo Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dia" % (len(df2),len(dias_train),len(dias_test)))
from sklearn.neural_network import MLPRegressor

# Treino da rede neural

mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=1000, shuffle=True)

mlp.fit(X=dias_train.values.reshape(-1,1),y=cases_train.values.ravel())
y_previsto = mlp.predict(dias_test.values.reshape(-1,1))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Estimador baseado em RNA MLP para o número de casos",

    xaxis_title="Dias desde a primeira notificação",

    yaxis_title="Quantidade de casos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3

    )

)



fig = go.Figure(data=[

    go.Scatter(x=dias_train, y=df2['cases'][:-1], name='Dados de Treinamento',mode="lines+markers"),

    go.Scatter(x=dias_test, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),

    go.Scatter(x=dias_test, y=df2['cases'][-1:], name = 'Casos Reais',mode="lines+markers")

])

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0= previsao + 0.5,

            y0=120,

            x1=previsao + 0.5,

            y1=max(df2['cases']) + 100,

            line=dict(

                width=1.5,

                dash= "dash"

            )

))



fig.add_trace(go.Scatter(

    x=[previsao - 0.5],

    y=[2],

    text=["Início da previsão"],

    mode="text",

))

fig['layout'].update(layout)



fig.show()
print("Erro Médio Absoluto: {0:6.3f} casos".format(mean_absolute_error(cases_test,y_previsto)))
import matplotlib.pyplot as plt

import statsmodels.api as sm



df3 = None

df3 = df2.copy()

df3.head()

df3.set_index('dias',inplace=True)

df3.drop(['date','deaths'],axis=1,inplace=True)
sm.graphics.tsa.plot_acf(df3.values.squeeze(), lags=10)

plt.show()
sm.graphics.tsa.plot_pacf(df3.values.squeeze(), lags=10)

plt.show()
df3['yesterday'] = df3['cases'].shift(1,fill_value=0)

df3.reset_index(level=0, inplace=True)

df3.head()
## Treino

dias_train = df3[['dias','yesterday']][:-1]

cases_train = df3['cases'][:-1]



## Teste

dias_test = df3[['dias','yesterday']][-1:]

cases_test =  df3['cases'][-1:]



previsao = len(df3) - len(dias_test)



print("Novo Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dia" % (len(df2),len(dias_train),len(dias_test)))
from sklearn.neural_network import MLPRegressor

# Treino da rede neural

mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=1000, shuffle=True)

mlp.fit(X=dias_train.values,y=cases_train.values.ravel())
y_previsto = mlp.predict(dias_test.values)
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "RNA MLP para previsão um dia à frente com dia anterior nos atributos",

    xaxis_title="Dias desde a primeira notificação",

    yaxis_title="Quantidade de casos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 4

    )

)



fig = go.Figure(data=[

    go.Scatter(x=dias_train['dias'], y=df3['cases'][:-1], name='Dados de Treinamento',mode="lines+markers"),

    go.Scatter(x=dias_test['dias'], y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),

    go.Scatter(x=dias_test['dias'], y=df3['cases'][-1:], name = 'Casos Reais',mode="lines+markers")

])

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0= previsao + 0.5,

            y0=120,

            x1=previsao + 0.5,

            y1=max(df2['cases']) + 100,

            line=dict(

                width=1.5,

                dash= "dash"

            )

))



fig.add_trace(go.Scatter(

    x=[previsao - 0.5],

    y=[2],

    text=["Início da previsão"],

    mode="text",

))

fig['layout'].update(layout)



fig.show()
print("Erro Médio Absoluto: {0:6.3f} casos".format(mean_absolute_error(cases_test,y_previsto)))
df3.tail()
from sklearn.neural_network import MLPRegressor



# Para uso nos dias que fiz previsão à posteriori

#df3.drop(df3.tail(1).index,inplace=True) 



tomorrow = max(df3['dias']) + 1

today_cases = df3.loc[df3['dias'] == max(df3['dias'])]['cases']



results = []

for i in range(20):



    # Treino da rede neural

    mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

    mlp.fit(X=df3[['dias','yesterday']].values,y=df3['cases'].values.ravel())

    



    x = pd.Series([tomorrow,int(today_cases)]).values.reshape(1,-1)

    tomorrow_cases = mlp.predict(x)

    results.append(tomorrow_cases)
tomorrow_data = (datetime.datetime.now()).strftime('%d/%m/%Y')

print("Previsão de casos para {0} no Brasil, a conferir na coletiva diária das 17h30min: {1}".format(tomorrow_data,int(min(results))))
y_true = [9056, 10278, 11130, 12056,14347, 15927, 17857, 19638, 20727, 22169, 23430, 25262, 28320, 30425, 33682, 36599, 38654, 40581, 43079, 45757, 52995, 58509, 63584, 66501, 71886, 78162, 85380, 91589, 96396,101147,107780, 114715,125218, 135106, 145328, 155939, 162699, 168331, 177589, 188974, 202918, 218223, 233142, 241080, 254220, 271628, 291579, 310087, 330890, 347398, 363211, 374898, 391222, 411821, 438238, 465166, 498440, 514849, 526447, 555393,584016, 614941]

y_previsto = [9152, 10432, 11766, 12521, 13379, 15316, 17943, 20071, 21923, 21919, 24097, 25353, 27010, 30788, 32730, 36598, 39721, 41664, 43517, 45985, 52806, 56585, 62838, 66246, 71186, 77052,83951, 91967,98552, 103467, 107909,111746,122221, 133920, 144682, 155719, 167089, 173487, 178559, 188156, 200292, 215428, 232085, 248059, 255226, 268995, 287681, 309425, 329081, 351431, 368255, 384151, 394788, 411194, 431782, 461557, 490163, 526460, 542052,551830,582685, 612518] 



print("Raiz do Erro Médio Quadrático (RMSE): {0:6.4f}".format(mean_squared_error(y_true,y_previsto)**0.5))

print("R2-Score: {0:6.4f}".format(r2_score(y_true,y_previsto)))
# Criação dos rótulos para o eixo x com datas

labels = []

start_date = datetime.date(2020, 4, 3)

end_date = datetime.date.today()

delta = datetime.timedelta(days=1)

while start_date < end_date:

    labels.append(start_date.strftime('%d/%m/%Y'))

    start_date += delta



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Visualizando previsões one-day-ahead realizadas com o modelo proposto",

    xaxis_title="Datas",

    yaxis_title="Quantidade de casos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3,

        tickangle = 295

    )

)



fig = go.Figure(data=[

    go.Scatter(x=labels, y=y_true, name='Casos Reais',mode="lines+markers"),

    go.Scatter(x=labels, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),

])



fig['layout'].update(layout)



fig.show()
residuos = []

datas = []

inicio = datetime.datetime.now() - datetime.timedelta(days=len(y_true))

for (x,y) in zip(y_true,y_previsto):

    r = (x-y)

    residuos.append(r)

    datas.append(inicio.strftime('%d/%m/%Y'))

    inicio += datetime.timedelta(days=1)
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title= "Visualização dos Resíduos",

    xaxis_title="Dia da Previsão",

    yaxis_title="Resíduos",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 1,

        dtick = 3,

        tickangle = 295

    )

)



fig = go.Figure(data=[

    go.Scatter(x=datas, y=residuos, name='Residuos',mode="markers")

])

fig.add_shape(

        # Horizontal Line

        dict(

            type="line",

            x0= 0,

            y0= 0,

            x1= len(y_true),

            y1=0,

            name = "Reta Zero",

            line=dict(

                width=3,

                dash= "dash"

            ),

            

))



fig.add_trace(go.Scatter(

    x=[len(y_true)],

    y=[300],

    text=["Reta Zero"],

    mode="text",

))

fig['layout'].update(layout)



fig.show()