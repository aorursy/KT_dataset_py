import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import requests
import io

offline.init_notebook_mode(connected=True)
df = pd.read_csv('../input/cotaparlamentar/cota_parlamentar_sp.csv')
df['datemissao'].fillna(method='ffill', inplace=True)
df['datemissao']=df['datemissao'].apply(lambda x : str(x)[6:10] + str(x)[2:5])
url = 'https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv'
s=requests.get(url).content
pokemons=pd.read_csv(io.StringIO(s.decode('utf-8')))
gastoes = df.groupby('sgpartido').size().nlargest(5).index.values
gasto = df.where(df['sgpartido'].apply(lambda x : str(x) in gastoes)).groupby(['sgpartido', 'datemissao']).sum()
data=[go.Scatter(x=gasto.loc[partido].index.values, y=gasto.loc[partido]['vlrdocumento'].values, name=partido) for partido in gastoes]
offline.iplot(data)
partidos = df.groupby(['sgpartido']).size()
data=[go.Bar(x=partidos.index.values, y=partidos.values)]
offline.iplot(data)
data = [go.Histogram(x=gasto['vlrdocumento'])]
offline.iplot(data)
attributes = ['HP','Attack','Sp. Atk','Defense','Sp. Def','Speed','HP']
data = [go.Scatterpolar(
    r=pokemons.loc[i][attributes].values,
    theta=attributes,
    name=pokemons.loc[i]['Name']
    ) 
        for i in pokemons.loc[pokemons['Legendary']==True].index
]
offline.iplot(data)
data = [go.Box(y=gasto.loc[partido]['vlrdocumento'], name=partido) for partido in gastoes]
offline.iplot(data)
gastoes = df.groupby('sgpartido').size().nlargest(5).index.values
gasto = df.where(df['sgpartido'].apply(lambda x : str(x) in gastoes)).groupby(['sgpartido', 'datemissao']).size()
data = [go.Violin(y=gasto.loc[partido], name=partido) for partido in gastoes]
offline.iplot(data)
