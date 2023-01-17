from datetime import datetime
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
df = pd.read_csv('../input/BTS_Spotify.csv')
df_release_date = pd.read_csv('../input/BTS_Album_Release.csv')
df_release_date = df_release_date.iloc[:,1:]
df.head()
df_release_date.head()
album_release = {row['index']: datetime.strptime(row['release_date'], "%Y-%m-%d")
                 for index, row in df_release_date.iterrows()}
categories = ['Danceability',
 'Energy',
 'Speechiness',
 'Acousticness',
 'Liveness',
 'Valence']
px.histogram(df, x = "Key")
mode = list(df['Mode'].value_counts().index)
size = list(df['Mode'].value_counts())

fig = px.pie(values=size, names = mode)
fig.update_layout(legend_title_text='Mode')
fig.show()
fig = make_subplots(rows=3, cols=2,
                    subplot_titles = ['Energy', 'Danceability', 'Speechiness',
                                      'Acousticness', 'Liveness', 'Valence'])
fig.add_trace(go.Histogram(x=df['Energy'], name = 'Energy'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['Danceability'], name = 'Danceability'), row=1, col=2)
fig.add_trace(go.Histogram(x=df['Speechiness'], name = 'Speechiness'), row=2, col=1)
fig.add_trace(go.Histogram(x=df['Acousticness'], name = 'Acousticness'), row=2, col=2)
fig.add_trace(go.Histogram(x=df['Liveness'], name = 'Liveness'), row=3, col=1)
fig.add_trace(go.Histogram(x=df['Valence'], name = 'Valence'), row=3, col=2)
fig.update_layout(title = "Distribution of Spotify metrics of BTS tracks")
fig.show()
album_summary = df.groupby("Album").mean().reset_index()
album_summary['date'] = album_summary['Album'].map(album_release)
album_summary = album_summary.sort_values(by = ['date'])
album_summary.head()
album = album_summary['Album']
fig = go.Figure()
fig.add_trace(go.Scatter(x = album, y = album_summary['Danceability'],
             mode = 'lines+markers', name = 'Avg. Danceability',))
fig.add_trace(go.Scatter(x = album, y = album_summary['Energy'],
             mode = 'lines+markers', name = 'Avg. Energy'))
fig.add_trace(go.Scatter(x = album, y = album_summary['Liveness'],
             mode = 'lines+markers', name = 'Avg. Liveness'))
fig.add_trace(go.Scatter(x = album, y = album_summary['Valence'],
             mode = 'lines+markers', name = 'Avg. Valence'))
fig.add_trace(go.Scatter(x = album, y = album_summary['Acousticness'],
             mode = 'lines+markers', name = 'Avg. Acousticness'))
fig.add_trace(go.Scatter(x = album, y = album_summary['Speechiness'],
             mode = 'lines+markers', name = 'Avg. Speechiness'))
fig = go.Figure()
album_names = album_release.keys()
for a in album_names:
    temp = df[df["Album"] == a]
    for index, row in temp.iterrows():    
        fig.add_trace(go.Scatterpolar(
              r= row[categories],
              theta=categories,
              fill='toself',
              name= row["TrackName"]
        ))
    fig.update_layout(title = "Album: " + a)
    fig.show()
    fig = go.Figure()
mots7 = df[df["Album"] == "MAP OF THE SOUL : 7"]

fig = go.Figure()
fig.add_trace(go.Scatter(x = mots7["TrackName"],
                         y = mots7['Danceability'],
                         name = 'Danceability',
                         mode = 'lines+markers'))
fig.add_trace(go.Scatter(x = mots7["TrackName"],
                         y = mots7['Energy'],
                         name = 'Energy',
                         mode = 'lines+markers'))
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = mots7["TrackName"],  y = mots7['Tempo'], mode = 'lines+markers'))
fig2.update_layout(yaxis_title = "Tempo")
fig2.show()
fig = go.Figure()
fig.add_trace(go.Bar(y = mots7['TrackName'], x = mots7['Energy'], name = 'Energy', orientation = "h"))
fig.add_trace(go.Bar(y = mots7['TrackName'], x = mots7['Danceability'], name = 'Danceability', orientation = "h"))
fig.add_trace(go.Bar(y = mots7['TrackName'], x = mots7['Valence'], name = 'Valence', orientation = "h"))
fig.update_layout(xaxis = dict(tickangle = 90), barmode='stack')
fig.show()