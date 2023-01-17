import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## Distribuição das faixas etárias

id_0_1 = 5681

id_1_4 = 23433

id_5_9 = 30761

id_10_14 = 33101

id_15_29 = 87267

id_39_49 = 70930

id_50_69 = 33226

id_70_100 = 10181



total_2010 = [id_0_1 , id_1_4 , id_5_9 , id_10_14 , id_15_29 , id_39_49 , id_50_69 , id_70_100]

total_2015 = 292520

faixa_etaria = ['Menor de 01 ano','01 à 04 anos','05 à 09 anos','10 à 14 anos ','15 à 29 anos','30 à 49 anos','50 à 69 anos ','70 anos e mais']
## Dataframe para realizar operações

dados = pd.DataFrame(total_2010, index=faixa_etaria,columns=['População 2010'])

dados['% População'] = dados['População 2010']/dados['População 2010'].sum()*100

dados['População 2015'] = round(total_2015 * dados['% População'] / 100)



print('Total Polução 2010 = ', dados['População 2010'].sum())

print('Total Polução 2015 = ', dados['População 2015'].sum())



dados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))

fig.suptitle('Distribuição Populacional ano 2015 - Santarém - Pará')

dados['População 2015'].plot.pie(ax=ax1,label='',autopct='%1.1f%%',shadow=True)

dados['População 2015'].plot.bar(label='',ax=ax2)

plt.show()
taxas = np.array([0,0,0,0.18,0.28,0.32,1.3,8])

taxas = taxas / 100



dados['Taxa Mortalidade'] = taxas
# 60% populacao infectada

dados['60%'] = dados['População 2015'] * 0.6

# 40% populacao infectada

dados['40%'] = dados['População 2015'] * 0.4

# 20% populacao infectada

dados['20%'] = dados['População 2015'] * 0.2

# 10% populacao infectada

dados['10%'] = dados['População 2015'] * 0.1

# 1% populacao infectada

dados['5%'] = dados['População 2015'] * 0.05

# 1% populacao infectada

dados['1%'] = dados['População 2015'] * 0.01
# Arredondamento para encontrar o número de possíveis mortes

dados['Mortes 60%'] = round(dados['60%'] * dados['Taxa Mortalidade'])

dados['Mortes 40%'] = round(dados['40%'] * dados['Taxa Mortalidade'])

dados['Mortes 20%'] = round(dados['20%'] * dados['Taxa Mortalidade'])

dados['Mortes 10%'] = round(dados['10%'] * dados['Taxa Mortalidade'])

dados['Mortes 5%'] = round(dados['5%'] * dados['Taxa Mortalidade'])

dados['Mortes 1%'] = round(dados['1%'] * dados['Taxa Mortalidade'])
#Remoção da coluna de população e percentual 2010

dados = dados.drop(['População 2010','% População'], axis = 1)
#Tabela para os dados

dados
colors = ['b', 'g', 'r', 'c', 'm', 'y']



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{"type": "domain"}, {"type": "domain"}],

           [{"type": "domain"}, {"type": "domain"}]],

)



fig.add_trace(go.Pie(values=dados['20%'],labels=dados.index,title='Número de pessoas infectadas para percentual de 20% de infectados', titlefont_size=20),

              row=1, col=1)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.add_trace(go.Pie(values=dados['10%'],labels=dados.index,title='Número de pessoas infectadas para percentual de 10% de infectados', titlefont_size=20),

              row=1, col=2)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.add_trace(go.Pie(values=dados['5%'],labels=dados.index,title='Número de pessoas infectadas para percentual de 5% de infectados', titlefont_size=20),

              row=2, col=1)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))



fig.add_trace(go.Pie(values=dados['1%'],labels=dados.index,title='Número de pessoas infectadas para percentual de 1% de infectados', titlefont_size=20),

              row=2, col=2)

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))



fig.update_layout(

    title="Número de infectados pelo percentual de espalhamento em Santarém/PA por grupos etários"

)



fig.update_layout(height=700,showlegend=True, titlefont_size=20)

fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{"type": "domain"}, {"type": "domain"}],

           [{"type": "domain"}, {"type": "domain"}]],

)



fig.add_trace(go.Pie(values=dados['Mortes 20%'],labels=dados.index,title='Letalidade para percentual 20% de infectados', titlefont_size=20),

              row=1, col=1)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.add_trace(go.Pie(values=dados['Mortes 10%'],labels=dados.index,title='Letalidade para percentual 10% de infectados', titlefont_size=20),

              row=1, col=2)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.add_trace(go.Pie(values=dados['Mortes 5%'],labels=dados.index,title='Letalidade para percentual 5% de infectados', titlefont_size=20),

              row=2, col=1)



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))



fig.add_trace(go.Pie(values=dados['Mortes 1%'],labels=dados.index,title='Letalidade para percentual 1% de infectados', titlefont_size=20),

              row=2, col=2)

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))



fig.update_layout(

    title="Letalidades pelo percentual de infectados em Santarém/PA por grupos etários"

)



fig.update_layout(height=700,showlegend=True, titlefont_size=20)

fig.show()
dados
fig = go.Figure()

fig.add_trace(go.Bar(x=dados.index,y=dados['20%'],name='20% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['10%'],name='10% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['5%'],name='5% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['1%'],name='1% Infectados'))



fig.update_traces(hoverinfo='y',  textfont_size=15, texttemplate='%{y}', textposition='inside')



fig.update_layout(

    title="Número infectados pelo Covid19 em Santarém/PA por grupos etários",

    yaxis_title="Número infectados",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.update_layout(showlegend=True)
fig = go.Figure()

fig.add_trace(go.Bar(x=dados.index,y=dados['Mortes 20%'],name='20% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['Mortes 10%'],name='10% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['Mortes 5%'],name='5% Infectados'))

fig.add_trace(go.Bar(x=dados.index,y=dados['Mortes 1%'],name='1% Infectados'))



fig.update_traces(hoverinfo='y',  textfont_size=15, texttemplate='%{y}', textposition='inside')



fig.update_layout(

    title="Número de letalidades pelo Covid19 em Santarém/PA por grupos etários",

    yaxis_title="Número letalidades",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)



fig.update_layout(showlegend=True)