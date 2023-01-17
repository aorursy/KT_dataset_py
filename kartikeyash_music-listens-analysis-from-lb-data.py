# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import json # load JSON data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import os



from datetime import datetime

from pandasql import sqldf # run SQL queries on dataframe

pysql = lambda q: sqldf(q, globals())



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
with open('../input/kartikeyaSh_lb-2019-03-20.json') as f:

    listens_json = json.load(f)

columns=['timestamp', 'artist_name', 'track_name', 'release_name']

data = [[listen.get(column) for column in columns ] for listen in listens_json]

data = [[idx] + listen for idx, listen in enumerate(data)]

data = pd.DataFrame(data, columns=['id'] + columns)

data['date_time'] = pd.to_datetime(data['timestamp'], unit='s')

# Convert time to my timezone

data['date_time'] = data['date_time'] + pd.Timedelta('5 hour 30 min')

data.info()
data.head()
top_tracks = pysql("""

SELECT

    id, track_name, count(track_name) AS track_count , artist_name

FROM

    data

GROUP BY

    track_name

ORDER BY

    count(track_name) DESC

LIMIT 10

""")



track_artist_name = ["{0}<br>By: {1}".format(track, artist) for (track, artist) in zip(top_tracks['track_name'], top_tracks['artist_name'])]



graph = [go.Bar(

            x=top_tracks['track_name'],

            y=top_tracks['track_count'],

            text=track_artist_name,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5,

                )

            ),

            opacity=0.7,

)]



layout = go.Layout(

    title='Top 10 Listens',

    plot_bgcolor='azure',

    paper_bgcolor='azure',

    titlefont=dict(

            size=20,

            color='black'

    ),

    xaxis=dict(

        title='Tracks',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='gray'

        )

    ),

    yaxis=dict(

        title='Listen Count',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='black'

        )

    ),

)



fig = go.Figure(data=graph, layout=layout)



iplot(fig)

top_artists = pysql("""

SELECT

    id, artist_name, count(artist_name) AS artist_count 

FROM

    data

GROUP BY

    artist_name

ORDER BY

    count(artist_name) DESC

LIMIT 10

""")



graph = [go.Bar(

            x=top_artists['artist_name'],

            y=top_artists['artist_count'],

            text=top_artists['artist_name'],

            marker=dict(

                color='purple',

                line=dict(

                    color='blue',

                    width=1.5,

                )

            ),

            opacity=0.6,

)]



layout = go.Layout(

    title='Top 10 Artists',

    plot_bgcolor='pink',

    paper_bgcolor='pink',

    titlefont=dict(

            size=20,

            color='black'

    ),

    xaxis=dict(

        title='Artists',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='gray'

        )

    ),

    yaxis=dict(

        title='Listen Count',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='black'

        )

    ),

)



fig = go.Figure(data=graph, layout=layout)



iplot(fig)
top_releases = pysql("""

SELECT

    id, release_name, count(release_name) AS release_count 

FROM

    data

WHERE

    release_name IS NOT NULL

GROUP BY

    release_name

ORDER BY

    count(release_name) DESC

LIMIT 10

""")



graph = [go.Bar(

            x=top_releases['release_name'],

            y=top_releases['release_count'],

            text=top_releases['release_name'],

            marker=dict(

                color='seagreen',

                line=dict(

                    color='green',

                    width=1.5,

                )

            ),

            opacity=0.6,

)]



layout = go.Layout(

    title='Top 10 Releases',

    plot_bgcolor='lightgreen',

    paper_bgcolor='lightgreen',

    titlefont=dict(

            size=20,

            color='black'

    ),

    xaxis=dict(

        title='Releases',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='gray'

        )

    ),

    yaxis=dict(

        title='Listen Count',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='black'

        )

    ),

)



fig = go.Figure(data=graph, layout=layout)



iplot(fig)
daily_listen_count =  pysql("""

SELECT

    strftime('%d-%m-%Y', date_time) as date, count(strftime('%d-%m-%Y', date_time)) AS listen_count

FROM

    data

GROUP BY

    strftime('%d-%m-%Y', date_time)

ORDER BY

    strftime('%Y-%m-%d', date_time)

""")



ts_min = daily_listen_count['date'].min()

ts_max = daily_listen_count['date'].max()



graph = [go.Scatter(

            x=daily_listen_count['date'],

            y=daily_listen_count['listen_count'],

            text=daily_listen_count['date'],

            marker=dict(

                color='blue',

                line=dict(

                    color='blue',

                    width=4,

                )

            ),

            opacity=0.6,

)]



layout = go.Layout(

    title='Daily Listens Count',

    plot_bgcolor='azure',

    paper_bgcolor='azure',

    titlefont=dict(

            size=20,

            color='black'

    ),

    xaxis=dict(

        title='Timeline',

        showticklabels=False,

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='gray'

        )

        

    ),

    yaxis=dict(

        title='Listens Count',

        showgrid=True,

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='black'

        )

    ),

)



fig = go.Figure(data=graph, layout=layout)

iplot(fig)
daily_timing =  pysql("""

SELECT

    strftime('%H', date_time) as hour, count(strftime('%H', date_time)) AS listen_count

FROM

    data

GROUP BY

    strftime('%H', date_time)

ORDER BY

    strftime('%H', date_time)

""")



graph = [go.Bar(

            x=daily_timing['hour'],

            y=daily_timing['listen_count'],

            marker=dict(

                color='blue',

                line=dict(

                    color='mediumorchid',

                    width=2.5,

                )

            ),

            opacity=0.6,

)]



layout = go.Layout(

    title='Daily Listening timings',

    plot_bgcolor='lightgreen',

    paper_bgcolor='lightgreen',

    titlefont=dict(

            size=20,

            color='black'

    ),

    xaxis=dict(

        title='Hours(24hrs format)',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='gray'

        ),

    ),

    yaxis=dict(

        title='Listens Count',

        showgrid=True,

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=12,

            color='black'

        )

    ),

)



fig = go.Figure(data=graph, layout=layout)

iplot(fig)
top_track_listen_timeline = []

for track in top_tracks['track_name']:

    if "'" in track:

        track = track.replace("'", "''")

    top_track_listen_timeline.append(pysql("""

        SELECT

            track_name, strftime('%Y-%m-%d', date_time) as date, timestamp,

            count(strftime('%d-%m-%Y', date_time)) AS listen_count

        FROM

            data

        WHERE

            track_name = '{0}'

        GROUP BY

            strftime('%d-%m-%Y', date_time)

        ORDER BY

            strftime('%Y-%m-%d', date_time)

    """.format(track))

    )



for idx, track in enumerate(top_tracks['track_name']):

    graph = [go.Scatter(

                x=top_track_listen_timeline[idx]['date'],

                y=top_track_listen_timeline[idx]['listen_count'],

                marker=dict(

                    color='blue',

                ),

                opacity=0.6,

    )]



    layout = go.Layout(

        title=track,

        plot_bgcolor='azure',

        paper_bgcolor='azure',

        titlefont=dict(

                size=20,

                color='black'

        ),

        xaxis=dict(

            title='Timeline',

            zerolinewidth=4,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )



        ),

        yaxis=dict(

            title='Listens Count',

            showgrid=True,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )

        ),

    )



    fig = go.Figure(data=graph, layout=layout)

    iplot(fig)
top_artist_listen_timeline = []

for artist in top_artists['artist_name']:

    if "'" in artist:

        artist = artist.replace("'", "''")

    top_artist_listen_timeline.append(pysql("""

        SELECT

            artist_name, strftime('%Y-%m-%d', date_time) as date, timestamp,

            count(strftime('%d-%m-%Y', date_time)) AS listen_count

        FROM

            data

        WHERE

            artist_name = '{0}'

        GROUP BY

            strftime('%d-%m-%Y', date_time)

        ORDER BY

            strftime('%Y-%m-%d', date_time)

    """.format(artist))

    )



for idx, artist in enumerate(top_artists['artist_name']):

    graph = [go.Scatter(

                hoverinfo='x+y+text',

                x=top_artist_listen_timeline[idx]['date'],

                y=top_artist_listen_timeline[idx]['listen_count'],

                text=artist,

                marker=dict(

                    color='red',

                ),

                opacity=0.6,

    )]



    layout = go.Layout(

        title=artist,

        plot_bgcolor='lightpink',

        paper_bgcolor='lightpink',

        titlefont=dict(

                size=20,

                color='black'

        ),

        xaxis=dict(

            title='Timeline',

            zerolinewidth=4,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )



        ),

        yaxis=dict(

            title='Listens Count',

            showgrid=True,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )

        ),

    )



    fig = go.Figure(data=graph, layout=layout)

    iplot(fig)

top_release_listen_timeline = []

for release in top_releases['release_name']:

    if "'" in release:

        release = release.replace("'", "''")

    top_release_listen_timeline.append(pysql("""

        SELECT

            release_name, strftime('%Y-%m-%d', date_time) as date, timestamp,

            count(strftime('%d-%m-%Y', date_time)) AS listen_count

        FROM

            data

        WHERE

            release_name = '{0}'

        GROUP BY

            strftime('%d-%m-%Y', date_time)

        ORDER BY

            strftime('%Y-%m-%d', date_time)

    """.format(release)

    ))



for idx, release in enumerate(top_releases['release_name']):

    graph = [go.Scatter(

                x=top_release_listen_timeline[idx]['date'],

                y=top_release_listen_timeline[idx]['listen_count'],

                marker=dict(

                    color='blue',

                ),

                opacity=0.6,

    )]



    layout = go.Layout(

        title=release,

        plot_bgcolor='lightgreen',

        paper_bgcolor='lightgreen',

        titlefont=dict(

                size=20,

                color='black'

        ),

        xaxis=dict(

            title='Timeline',

            zerolinewidth=4,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )



        ),

        yaxis=dict(

            title='Listens Count',

            showgrid=True,

            titlefont=dict(

                size=16,

                color='black'

            ),

            tickfont=dict(

                size=12,

                color='black'

            )

        ),

    )



    fig = go.Figure(data=graph, layout=layout)

    iplot(fig)