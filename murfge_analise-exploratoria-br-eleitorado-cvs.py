import os

import json
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import pandas_profiling
df = pd.read_csv('../input/br-eleitorado/br_eleitorado.csv', delimiter=",")

df.head(3)
print(df.shape)
print(df.columns)
df.info()
df['total_eleitores'].sum()
df.describe()
import folium

from plotly.offline import plot

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import plotly.express as px
corrArr = df.corr()

fig = go.Figure(data=go.Heatmap(z=corrArr, x=corrArr.columns, y=corrArr.columns))

fig.update_layout(

    title={

        'text':'Matriz de correlação',

        'font':dict(size=24, color="#777779")

    })

fig.show()
df = df.drop(columns=['cod_municipio_tse'])
arrQtdEleitores = df['uf'].value_counts().reset_index()

arrQtdEleitores.columns = ['uf', 'qtd_eleitores']

figQtdEleitores = px.bar(arrQtdEleitores, x='uf', y='qtd_eleitores', color='qtd_eleitores')

figQtdEleitores.update_layout(title={'text':'Qtd. municipios por estado', 'font':dict(size=24, color="#777779")})

figQtdEleitores.show()
arrSumEleitores = df.groupby('uf')['total_eleitores'].sum().reset_index().sort_values(by=['total_eleitores'], ascending=[False])

figSumEleitores = px.bar(arrSumEleitores, x='uf', y='total_eleitores', color='total_eleitores')

figSumEleitores.update_layout(title={'text':'Total de eleitores por estado', 'font':dict(size=24, color="#777779")})

figSumEleitores.show()
mesh_data = json.load(open('../input/br-geojson/br_geoJson.json'))

states_data = json.load(open('../input/estadosjson/estados.json'))



states_name = [estados["nome"] for estados in states_data]

states_id = [estados["id"] for estados in states_data]

state_initials = [estados["sigla"] for estados in states_data]



states = pd.DataFrame.from_dict({'id':states_id,'name':states_name,"uf":state_initials})



final_data = pd.merge(left = states,right = arrSumEleitores,on="uf",how="outer")

final_data.set_index('id', inplace=True)

final_data.fillna(0, inplace=True)

final_data.loc[:,'total_eleitores'] = final_data.loc[:,'total_eleitores'] / 1000000



for state in mesh_data['features']:

    codarea = state['properties']['codarea']

    state['properties']['uf'] = str(final_data.loc[int(codarea), "uf"])

    state['properties']['total_eleitores'] = str(final_data.loc[int(codarea), "total_eleitores"])

    

bins = np.linspace(final_data['total_eleitores'].min(), final_data['total_eleitores'].max(), 10).tolist()

federal_district = [-15.7757875, -48.0778477]
mapaPorEstado = folium.Map(

    location=federal_district,

    zoom_start=4

)



folium.Choropleth(

    geo_data=mesh_data,

    data=final_data,

    columns=['uf', 'total_eleitores'],

    key_on='feature.properties.uf',

    fill_color='Reds',

    fill_opacity=0.7,

    line_opacity=0.5,

    legend_name='Eleitores por estado (x 1M)',

    bins=bins

).add_to(mapaPorEstado)



mapaPorEstado
df.loc[df['uf'].isin(['PR','RS','SC']), 'region'] = 'sul'

df.loc[df['uf'].isin(['SP','ES','MG','RJ']), 'region'] = 'sudeste'

df.loc[df['uf'].isin(['MT','MS','GO']), 'region'] = 'centroeste'

df.loc[df['uf'].isin(['AM','RR','AP','PA','TO','RO','AC']), 'region'] = 'norte'

df.loc[df['uf'].isin(['MA','PI','CE','RN','PE','PB','SE','AL','BA']), 'region'] = 'nordeste'
final_data_reg = final_data.copy().reset_index().merge(df.groupby('uf')['region'].agg(lambda x: x.unique()).reset_index() ,

                                            how='outer',

                                            on='uf').set_index('id')



final_data_reg.loc[final_data_reg['uf'] == 'DF', 'region'] = 'centroeste'



for reg in final_data_reg.groupby('region').groups.keys():

    sum_eleitores = final_data_reg[final_data_reg['region'] == reg]['total_eleitores'].sum()

    final_data_reg.loc[final_data_reg['region'] == reg, 'sum_eleitores'] = sum_eleitores



for state in mesh_data['features']:

    codarea = state['properties']['codarea']

    state['properties']['region'] = str(final_data_reg.loc[int(codarea), "region"])

    state['properties']['sum_eleitores'] = str(final_data_reg.loc[int(codarea), "sum_eleitores"])

    

bins = np.linspace(final_data_reg['sum_eleitores'].min(), final_data_reg['sum_eleitores'].max(), 10).tolist()
mapaPorRegiao = folium.Map(

    location=federal_district,

    zoom_start=4

)



folium.Choropleth(

    geo_data=mesh_data,

    data=final_data_reg,

    columns=['region', 'sum_eleitores'],

    key_on='feature.properties.region',

    fill_color='Reds',

    fill_opacity=0.7,

    line_opacity=0.5,

    legend_name='Eleitores por estado (x 1M)',

    bins=bins

).add_to(mapaPorRegiao)



mapaPorRegiao
figGrpRegiao = px.bar(df.groupby('region')['total_eleitores'].sum().reset_index().sort_values('total_eleitores', ascending=False), 

                      x='region', 

                      y='total_eleitores', 

                      color='total_eleitores')

figGrpRegiao.update_layout(title={'text':'Total de eleitores por região', 'font':dict(size=24, color="#777779")})

figGrpRegiao.show()
labels = []

parents=[]

values=[]



for reg in df.groupby('region').groups.keys():

    labels.append(reg)

    parents.append("Brasil")

    values.append(df[df['region'] == reg]['total_eleitores'].sum())

    for est in df[df['region'] == reg].groupby('uf').groups.keys():

        labels.append(est)

        parents.append(reg)

        values.append(df[df['uf'] == est]['total_eleitores'].sum())

        grp_mun = df[df['uf'] == est].groupby('nome_municipio')['total_eleitores'].sum().reset_index().sort_values('total_eleitores', ascending=False)['nome_municipio'][:10]

        for mun in grp_mun:

            labels.append(mun)

            parents.append(est)

            values.append(df[(df['region'] == reg) & (df['uf'] == est) & (df['nome_municipio'] == mun)]['total_eleitores'].sum())
fig4 = px.treemap(

    title={"text":"Matriz Árvore - Total de eleitores por estado", 'font':dict(size=24, color="#777779")},

    names = labels,

    parents = parents,

    values = values,

    branchvalues = "total",

    maxdepth=3

)



fig4.show()
grp = df.groupby(['region', 'uf'])['gen_feminino', 'gen_masculino', 'gen_nao_informado', 'total_eleitores'].sum().reset_index()
figBoxPorRegiao = px.box(grp, x="region", y="total_eleitores")

figBoxPorRegiao.update_layout(title={'text':'Box Plot - Distribuição de eleitores por região', 'font':dict(size=24, color="#777779")})

figBoxPorRegiao.show()
faixa_idade = ['f_16', 'f_17', 'f_18_20', 'f_21_24', 'f_25_34', 'f_35_44', 'f_45_59', 'f_60_69', 'f_70_79', 'f_sup_79']



figHistBrasil = px.histogram(x=faixa_idade, 

                             y=df[df.columns.intersection(faixa_idade)].sum(axis=0), 

                             histfunc="sum", 

                             marginal="box")



figHistBrasil.update_layout(

    title={'text':'Histograma - Faixa etária Brasil', 'font':dict(size=24, color="#777779")},

    xaxis_title_text='faixa_idade',

    yaxis_title_text='total_eleitores'

)



figHistBrasil.show()



figHistPorEstado = make_subplots(rows=2, cols=3, subplot_titles=("Histograma Sul", 

                                                                 "Histograma Sudeste", 

                                                                 "Histograma Centroeste", 

                                                                 "Histograma Norte", 

                                                                 "Histograma Nordeste"))



figHistPorEstado.add_trace(go.Histogram(histfunc="sum", 

                                        y=df[df['region'] == 'sul'][(df.columns.intersection(faixa_idade))].sum(axis=0), 

                                        x=faixa_idade, name="Sul"), 

                           row=1, col=1)



figHistPorEstado.add_trace(go.Histogram(histfunc="sum", 

                                        y=df[df['region'] == 'sudeste'][(df.columns.intersection(faixa_idade))].sum(axis=0), 

                                        x=faixa_idade, 

                                        name="Sudeste"), 

                           row=1, col=2)



figHistPorEstado.add_trace(go.Histogram(histfunc="sum", 

                                        y=df[df['region'] == 'centroeste'][(df.columns.intersection(faixa_idade))].sum(axis=0), 

                                        x=faixa_idade, 

                                        name="Centroeste"), 

                           row=1, col=3)



figHistPorEstado.add_trace(go.Histogram(histfunc="sum", 

                                        y=df[df['region'] == 'norte'][(df.columns.intersection(faixa_idade))].sum(axis=0), 

                                        x=faixa_idade, 

                                        name="Norte"), 

                           row=2, col=1)



figHistPorEstado.add_trace(go.Histogram(histfunc="sum", 

                                        y=df[df['region'] == 'nordeste'][(df.columns.intersection(faixa_idade))].sum(axis=0), 

                                        x=faixa_idade, 

                                        name="Nordeste"), 

                           row=2, col=2)



figHistPorEstado.update_yaxes(title_text="total_eleitores", row=1, col=1)

figHistPorEstado.update_yaxes(title_text="total_eleitores", row=2, col=1)
dfPercent = pd.DataFrame(columns=['region','faixa','%'])

for reg in df.groupby('region').groups.keys():

    for idade in faixa_idade:

        rown = len(dfPercent)

        dfPercent.loc[rown, 'region'] = reg

        dfPercent.loc[rown, 'faixa'] = idade

        dfPercent.loc[rown, '%'] = df[df['region'] == reg][idade].sum()

    



grpPercent = dfPercent.groupby('region')['%'].sum().reset_index()



for index, row in dfPercent.iterrows():

    v = grpPercent.loc[grpPercent['region'] == row['region'], '%'].values[0]

    dfPercent.loc[index, '%'] = (dfPercent.loc[index, '%'] / v) * 100

    

figGaussPercent = px.line(dfPercent, x="faixa", y="%", color='region')

figGaussPercent.update_layout(title={'text':'Valores normalizados por região', 'font':dict(size=24, color="#777779")})

figGaussPercent.show()
figHistGrpd = px.bar(dfPercent, x="faixa", y="%", color='region', barmode='group')

figHistGrpd.update_layout(title={'text':'Histograma agrupado por região', 'font':dict(size=24, color="#777779")})

figHistGrpd.show()
grpGen = df.groupby(['region', 'uf'])[['gen_masculino', 'gen_feminino', 'gen_nao_informado']].sum().reset_index()

grpGemM = grpGen.drop(['gen_feminino', 'gen_nao_informado'], axis=1)

grpGemM.columns = ['region','uf', 'eleitores']

grpGemM['genero'] = 'masculino'



grpGemF = grpGen.drop(['gen_masculino', 'gen_nao_informado'], axis=1)

grpGemF.columns = ['region','uf', 'eleitores']

grpGemF['genero'] = 'feminino'



grpGemNi = grpGen.drop(['gen_masculino', 'gen_feminino'], axis=1)

grpGemNi.columns = ['region','uf', 'eleitores']

grpGemNi['genero'] = 'nao informado'



frames = [grpGemM, grpGemF, grpGemNi]



dfGen = pd.concat(frames).reset_index().drop('index', axis=1).sort_values(by=['eleitores'], ascending=[False])
#lsrRegiao=list(dfGen.groupby('region').groups.keys())

lsrRegiao = dfGen.groupby('region').sum().reset_index().sort_values(by=['eleitores'], ascending=[False])['region']

figGenGrpdReg = go.Figure(data=[

    go.Bar(name='gênero=masculino', x=lsrRegiao, y=dfGen[dfGen['genero'] == 'masculino']['eleitores']),

    go.Bar(name='gênero=feminino', x=lsrRegiao, y=dfGen[dfGen['genero'] == 'feminino']['eleitores']),

    go.Bar(name='gênero=não informado', x=lsrRegiao, y=dfGen[dfGen['genero'] == 'nao informado']['eleitores'])

])

figGenGrpdReg.update_layout(

    title={'text':'Distribuição de gênero por região', 'font':dict(size=24, color="#777779")},

    yaxis=dict(title='Total de eleitores'),

    legend=dict(

        x=1.0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group'

)

figGenGrpdReg.show()
lsrEstado=dfGen.groupby('uf').sum().reset_index().sort_values(by=['eleitores'], ascending=[False])['uf']

figGenGrpdEst = go.Figure(data=[

    go.Bar(name='gênero=masculino', x=lsrEstado, y=dfGen[dfGen['genero'] == 'masculino']['eleitores']),

    go.Bar(name='gênero=feminino', x=lsrEstado, y=dfGen[dfGen['genero'] == 'feminino']['eleitores']),

    go.Bar(name='gênero=não informado', x=lsrEstado, y=dfGen[dfGen['genero'] == 'nao informado']['eleitores'])

])

figGenGrpdEst.update_layout(

    title={'text':'Distribuição de gênero por estado', 'font':dict(size=24, color="#777779")},

    yaxis=dict(title='Total de eleitores'),

    legend=dict(

        x=1.0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group'

)

figGenGrpdEst.show()
x = []

y_m = []

y_f = []

for index, row in grp.iterrows():

    x.append(row['region'])

    y_m.append(row['gen_masculino'])

    y_f.append(row['gen_feminino'])
figDispEstado = go.Figure()



figDispEstado.add_trace(go.Scatter(

    x=dfGen[dfGen['genero'] == 'masculino']['eleitores'], 

    y=lsrEstado,

    name='gênero=masculino',

    mode='markers',

    marker_color='#009add'

))



figDispEstado.add_trace(go.Scatter(

    x=dfGen[dfGen['genero'] == 'feminino']['eleitores'], 

    y=lsrEstado,

    name='gênero=feminino',

    mode='markers',

    marker_color='#FF4136'

))



figDispEstado.add_trace(go.Scatter(

    x=dfGen[dfGen['genero'] == 'nao informado']['eleitores'], 

    y=lsrEstado,

    name='gênero=não informado',

    mode='markers',

))



figDispEstado.update_layout(

    title={'text':'Dispersão - Distribuição de gênero por estado', 'font':dict(size=24, color="#777779")},

    yaxis=dict(title='Estados'),

    xaxis=dict(title='Total de eleitores'),

    legend=dict(

        x=1.0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    height=700

)



figDispEstado.update_traces(mode='markers', marker_line_width=1, marker_size=8)



figDispEstado.show()
figBoxPltGrp = go.Figure()



figBoxPltGrp.add_trace(go.Box(

    y=y_m,

    x=x,

    name='gênero=masculino',

    marker_color='#009add'

))



figBoxPltGrp.add_trace(go.Box(

    y=y_f,

    x=x,

    name='gênero=feminino',

    marker_color='#FF4136'

))



figBoxPltGrp.update_layout(

    yaxis_title='Total de eleitores',

    boxmode='group'

)



figBoxPltGrp.update_layout(title={'text':'Boxplot - Distribuição de eleitores por gênero', 'font':dict(size=24, color="#777779")})



figBoxPltGrp.show()