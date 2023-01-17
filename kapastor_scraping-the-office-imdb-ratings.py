import pandas as pd 

import urllib.request

from bs4 import BeautifulSoup

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



data_set = []

data_set_season = {}

for s in range(1,10):

    with urllib.request.urlopen('https://www.imdb.com/title/tt0386676/episodes?season='+str(s)) as response:

        html = response.read()

        season_set = []

        soup = BeautifulSoup(html)

        episodes = soup.findAll("div", {"class": "list_item"})

        for episode in episodes:

            title = episode.findAll("a", {"itemprop": "name"})

            airdate = episode.findAll("div", {"class": "airdate"})

            rating = episode.findAll("span", {"class": "ipl-rating-star__rating"})

            num_votes = episode.findAll("span", {"class": "ipl-rating-star__total-votes"})

            description = episode.findAll("div", {"class": "item_description"})

            row_data = [title[0].text,airdate[0].text,rating[0].text,num_votes[0].text.replace('(','').replace(')','').replace(',',''),description[0].text]

            row_data = [r.replace('\n','').strip() for r in row_data]

            data_set.append(row_data)

            season_set.append(row_data)

    data_set_season['Season'+str(s)]= season_set









df = pd.DataFrame(data_set,columns=['Title','AirDate','Rating','Num_Votes','Description'])

df.to_csv('TheOfficeIMDBPerEpisode.csv',index=False)
fig = go.Figure()

for s in range(1,10):

    df = pd.DataFrame(data_set_season['Season'+str(s)],columns=['Title','AirDate','Rating','Num_Votes','Description'])

    trace = go.Scatter(

                        x = df.AirDate,

                        y = df.Rating,

                        mode = "lines",

                        name = "Rating",

                        line=dict(color='rgb(67,67,67)', width=s))

    fig.add_trace(trace)



layout = dict(title = 'The Office IMDB Ratings Per Episode',

              xaxis= dict(title= 'Air Date',ticklen= 10,zeroline= False))

             



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    showlegend=False,

    plot_bgcolor='white'

)



iplot(fig)