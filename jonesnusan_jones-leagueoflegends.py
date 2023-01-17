# declarações

import pandas as pd

import numpy as np

# import matplotlib.pyplot as plt

import plotly.graph_objects as go

# from skimage.io import imread
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

semtag = dados.loc[dados['blueTeamTag']=='SemTag']

semtag[['League','Type','blueTeamTag','redTeamTag','blueTop','blueJungle','blueMiddle','blueADC','blueSupport','redTop','redJungle','redMiddle','redADC','redSupport']]
dados2015 = dados.loc[dados['Year'] == 2015]

dados2016 = dados.loc[dados['Year'] == 2016]

dados2017 = dados.loc[dados['Year'] == 2017]

dados2018 = dados.loc[dados['Year'] == 2018]

q1_2015 = dados2015[['bResult','rResult']].sum()

q1_2016 = dados2016[['bResult','rResult']].sum()

q1_2017 = dados2017[['bResult','rResult']].sum()

q1_2018 = dados2018[['bResult','rResult']].sum()
trace2015 = go.Bar( x = ['BlueTeam', 'RedTeam'],

                    y = [q1_2015.bResult, q1_2015.rResult],

                    name = '2015')

trace2016 = go.Bar( x = ['BlueTeam', 'RedTeam'],

                    y = [q1_2016.bResult, q1_2016.rResult],

                    name = '2016')

trace2017 = go.Bar( x = ['BlueTeam', 'RedTeam'],

                    y = [q1_2017.bResult, q1_2017.rResult],

                    name = '2017')

trace2018 = go.Bar( x = ['BlueTeam', 'RedTeam'],

                    y = [q1_2018.bResult, q1_2018.rResult],

                    name = '2018')

q1data = [trace2015, trace2016, trace2017, trace2018]



q1layout = go.Layout({

    'title' : {

        'text': 'Taxa de vitórias do lado azul e vermelho por ano',

        'font': {

            'size': 20

        }

    }

})



q1fig = go.Figure(data=q1data, layout=q1layout)



q1fig.show()
topchampb = dados['blueTopChamp'].value_counts()

topchampr = dados['redTopChamp'].value_counts()

topchampb.combine(topchampr, lambda x,y : x+y, fill_value=0)

topchamp = topchampb.sort_values(ascending=False)
champlist = []

for champ in topchamp.index:

    champlist.append(champ)



champselct = []

for champ in topchamp:

    champselct.append(champ)

tracechamp = go.Bar( x = champlist,

                    y = champselct)

q2layout = go.Layout({

    'title' : {

        'text': 'Campeões do Top mais escolhidos',

        'font': {

            'size': 20

        }

    }

})

q2data = [tracechamp]

q2fig = go.Figure(data=q2data, layout=q2layout)



q2fig.show()
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