import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding="ISO-8859-1")
df.head()
df.info()
temp = df[['Artist.Name', 'Popularity']].groupby('Artist.Name', as_index=False).mean().sort_values(by='Popularity', ascending=False)[:10]

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Artist.Name'], y=temp['Popularity'], text=temp['Popularity'], textposition='auto'))

fig.update_layout(title='Qual é a média da popularidade que um artista de topo tem?')

fig.show()
temp = df[['Artist.Name', 'Genre']].groupby(['Artist.Name'], as_index=False).count().sort_values(by='Genre', ascending=False)[:3]

fig = go.Figure(data=[

    go.Bar(x=temp['Artist.Name'], y=temp['Genre'])

])

fig.update_layout(title='Qual artista concentra a maior quantidade de música no top 50?')

fig.show()
temp = df[['Genre', 'Liveness']].groupby(['Genre'], as_index=False).count().sort_values(by='Liveness', ascending=False)[:9]

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Genre'], y=temp['Liveness']))

fig.update_layout(title='Qual o gênero concetra a maior quantidade de musica no top 50?')

fig.show()
temp = df[['Artist.Name', 'Energy']].groupby('Artist.Name', as_index=False).mean().sort_values(by='Energy', ascending=False)[:10]

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Artist.Name'], y=temp['Energy'], text=temp['Energy'], textposition='auto'))

fig.update_layout(title='Qual é a média de energia que um artista de topo tem?')

fig.show()
temp = df[['Artist.Name', 'Acousticness..']].groupby('Artist.Name', as_index=False).mean().sort_values(by='Acousticness..', ascending=False)[:10]

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Artist.Name'], y=temp['Acousticness..'], text=temp['Acousticness..'], textposition='auto'))

fig.update_layout(title='Qual é a média de acústica que um artista de topo tem?')

fig.show()
temp = df[['Artist.Name', 'Valence.']].groupby('Artist.Name', as_index=False).mean().sort_values(by='Valence.', ascending=False)[:10]

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Artist.Name'], y=temp['Valence.'], text=temp['Valence.'], textposition='auto'))

fig.update_layout(title='Qual é a média de humor positivo que um artista de topo tem?')

fig.show()
temp = df[['Artist.Name', 'Popularity', 'Energy', 'Acousticness..', 'Valence.']].groupby('Artist.Name', as_index=False).mean().sort_values(by='Popularity', ascending=False) [:20]



media = ['Popularity', 'Energy', 'Acousticness..', 'Valence.']



fig = go.Figure()



for me in media:

    fig.add_trace(go.Bar(x=temp['Artist.Name'], y=temp[me], name=me))

    

fig.update_layout(title='Conclusão do Top 20', barmode='stack', )