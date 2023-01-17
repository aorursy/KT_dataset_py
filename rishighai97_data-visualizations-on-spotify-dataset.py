import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

from datetime import datetime
dataset = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding = "ISO-8859-1")

dataset["year"] = dataset["year"].map(lambda x: datetime.strptime(str(x),'%Y').year)

display(dataset.head())
# Popularity of a particular genre over the years

genre = 'edm'

group_by_genre = dataset[dataset["top genre"]==genre].groupby("year").mean().sort_values(["pop"])

fig = go.Figure(

    data=[go.Bar(x=group_by_genre.index,y=group_by_genre["pop"])],

    layout_title_text="Popularity of genre - "+genre+" over the years"

)

fig.show()
# Genre ranking for a year

year = 2018

group_by_genre = dataset[dataset["year"]==year].groupby("top genre").mean().sort_values(["pop"])

fig = go.Figure(go.Bar(

    x = group_by_genre.index,

    y = group_by_genre["pop"]

),

layout_title_text="Genre ranking for the year- "+str(year)

)

fig.show()
# Genre ranking over the years

data = dataset.groupby("top genre").mean().sort_values(["pop"])

fig = go.Figure(

    data=[go.Bar(x=data.index,y=data["pop"])],

    layout_title_text="Genre ranking over the years"

)

fig.show()
# popularity of an artist over the years

artist = 'Justin Bieber'

artist_data = dataset[dataset["artist"]==artist].groupby("year").mean()

fig = go.Figure([go.Scatter(x=artist_data.index, y=artist_data["pop"])],layout_title_text="Popularity of "+artist+" over the years")

fig.show()
# Artist ranking for a year

year = 2017

data = dataset[dataset["year"]==year].groupby("artist").mean().sort_values(["pop"])

fig = go.Figure(

    data=[go.Bar(x=data.index,y=data["pop"])],

    layout_title_text="Artist ranking for the year - "+str(year)

)

fig.show()
# Artist ranking along with the number of hit songs released over the years

data = dataset.groupby("artist").mean().sort_values(["pop"])

data["artist"] = data.index.values

data["count"] = dataset.groupby("artist").count()["pop"].values



fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(go.Bar(x=data["artist"],

                 y=data["pop"],

                 text = data["count"],

                 hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{y:.2f}<br><b> # Songs: </b>%{text}',

                 showlegend = False

                ),

    secondary_y=False,

)



fig.add_trace(go.Scatter(

    x = data["artist"],

    y = data["count"],

    text = data["pop"],

    hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{text:.2f}<br><b> # Songs: </b>%{y}',

    showlegend = False),

    secondary_y=True,)



# Set x-axis title

fig.update_xaxes(title_text="Artist ranking along with the number of songs released over the years")



# Set y-axes titles

fig.update_yaxes(title_text="<b>Popularity</b>", secondary_y=False)

fig.update_yaxes(title_text="<b># Hit songs", secondary_y=True)



fig.show()
# Hit songs ranking for a year

year = 2019

data = dataset[dataset["year"]==year].sort_values(by=["pop"])

fig = go.Figure(

    data=[go.Bar(x=data["title"],y=data["pop"])],

    layout_title_text="Hit songs ranking for the year - "+str(year)

)

fig.show()
# Number of hit songs released by artists on spotify over the years

data = dataset.groupby("artist").count().sort_values(["pop"])

fig = go.Figure(

    data=[go.Bar(x=data.index,y=data["pop"])],

    layout_title_text="Number of hit songs released by artists on spotify over the years"

)

fig.show()