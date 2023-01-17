import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])
iplot([go.Histogram2dContour(x=reviews.head(500)['points'], 
                             y=reviews.head(500)['price'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])
df = reviews.assign(n=0).groupby(['points', 'price'])['n'].count().reset_index()
df = df[df["price"] < 100]
v = df.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=v)])
df = reviews['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text=df.index,
    z=df.values
)])
import pandas as pd
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)
iplot([go.Scatter(x=pokemon['Attack'], y=pokemon['Defense'], mode='markers')])
iplot([go.Histogram2dContour(x=pokemon['Attack'], 
                             y=pokemon['Defense'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=pokemon['Attack'], y=pokemon['Defense'], mode='markers')])
df = pokemon.assign(n=0).groupby(['Type 1', 'Type 2'])['n'].count().reset_index()
df
v = df.pivot(index='Type 1', columns='Type 2', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=v)])

df = pokemon.assign(n=0).groupby(['Attack', 'Defense'])['n'].count().reset_index()
df
v = df.pivot(index='Attack', columns='Defense', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=v)])

