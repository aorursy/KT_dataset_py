# declarações

import pandas as pd

import numpy as np

import plotly.graph_objects as go
# carregando dataset

dados = pd.read_csv('../input/leagueoflegends/LeagueofLegends.csv')
dados.shape
dados.describe()
dados.info()
dados.isnull().sum()
# Substituindo nomes dos times nulos por 'SemTag'

dados.update(dados[['blueTeamTag','redTeamTag']].fillna('SemTag'))
# Substituindo nomes dos jogadores nulos por 'SemNome'

dados.update(dados[['blueTop','blueJungle','blueMiddle','blueADC','blueSupport','redTop','redJungle','redMiddle','redADC','redSupport']].fillna('SemNome'))
# Por alguma razão alguns jogaores estavam com '=' no nome, foi substituido por 'SemNome'

dados.update(dados[['blueJungle','redJungle']].replace('=','SemNome'))
# Visualizando dados

##dados.loc[dados['blueTeamTag']=='SemTag'][['League','Type','blueTeamTag','redTeamTag','blueTop','blueJungle','blueMiddle','blueADC','blueSupport','redTop','redJungle','redMiddle','redADC','redSupport']]
dados[['Year', 'bResult', 'rResult']].groupby('Year', as_index=False).sum()
# excluindo dados que não serão utilizados

dados.drop(dados[dados.Year == 2014].index, inplace=True)

dados.drop(dados[dados.Year == 2018].index, inplace=True)
temp = dados[['Year', 'bResult', 'rResult']].groupby('Year', as_index=False).sum()

temp['total'] = temp[['bResult', 'rResult']].sum(axis=1)

temp['bPercent'] = (temp['bResult'] * 100)/ temp['total']

temp['rPercent'] = (temp['rResult'] * 100)/ temp['total']



fig = go.Figure(data=[

    go.Bar(name='Blue team', x=temp['Year'], y=temp['bPercent'], text=temp['bPercent'], textposition='auto'),

    go.Bar(name='Red team', x=temp['Year'], y=temp['rPercent'], text=temp['rPercent'], textposition='auto')

])

fig.update_layout(barmode='stack')

fig.show()
dados2017 = dados.loc[dados['Year'] == 2017]

bTopChamp = dados2017.loc[dados2017['bResult'] == 1]['blueTopChamp'].value_counts().to_frame()

rTopChamp = dados2017.loc[dados2017['rResult'] == 1]['redTopChamp'].value_counts().to_frame()

topChamp = bTopChamp.join(rTopChamp)

topChamp['total'] = topChamp['blueTopChamp'] + topChamp['redTopChamp']

topchamp = topChamp.sort_values(by=['total'], ascending=False)



fig = go.Figure([go.Bar(x=topChamp.index, y=topChamp['total'])])

fig.show()

leagues = dados['League'].unique().tolist()
leagues.remove('RR')

leagues.remove('IEM')

leagues.remove('MSI')

leagues.remove('WC')
gamelength = []

for l in leagues:

    df = dados.loc[dados["League"]==l]

    gamelength.append(df['gamelength'].mean())

tracegamelength = go.Bar( x = leagues,

                    y = gamelength)

q3layout = go.Layout({

    'title' : {

        'text': 'Média de Partidas por região',

        'font': {

            'size': 20

        }

    }

})

q3data = [tracegamelength]

q3fig = go.Figure(data=q3data, layout=q3layout)



q3fig.show()