!pip install pandas-bokeh
!pip install chart_studio
!pip install bar_chart_race
import altair as alt
import bar_chart_race as bcr
import pandas as pd
import pandas_profiling 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import chart_studio.plotly as py

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

from plotly.offline import iplot

from pandas_profiling import ProfileReport

PLOT_BGCOLOR='#99ff66'
episode = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')
anchor_t = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')
yt_t = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')
eps_desc = pd.read_csv('/kaggle/input/chai-time-data-science/Description.csv')
print(episode.shape[0])
episode['youtube_watch_hours'].sum()
hours = episode['youtube_watch_hours'].sum()
days = episode['youtube_watch_hours'].sum()/24
tot_views = episode['youtube_views'].sum()
tot_impressions = episode['youtube_impressions'].sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    title = 'Total Youtube Impressions',
    mode = "number",
    value = tot_impressions,
    domain = {'row': 0, 'column': 0}))


fig.add_trace(go.Indicator(
    title = "Total youtube views",
    mode = "number",
    value = tot_views,
    domain = {'row': 0, 'column': 1}))


fig.add_trace(go.Indicator(
    title = 'Total days of content watched',
    mode = "number",
    value = days,
    domain = {'row': 1, 'column': 0}))

fig.add_trace(go.Indicator(
    title = 'Total number of episodes',
    mode = "number",
    value = episode.shape[0],
    domain = {'row': 1, 'column': 1}))

fig.update_layout(width=700,height=400,title='<b>Chai Time Data Science Stats</b>',
                  template='seaborn',margin=dict(t=60,b=10,l=10,r=10),
                  grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},paper_bgcolor=PLOT_BGCOLOR)
ctd_pivot = pd.pivot_table(episode, index='release_date', values='youtube_views')
bcr.bar_chart_race(df = ctd_pivot, title = "Youtube views over time", cmap='dark12', figsize=(5, 3),)
top_yt_views = episode.sort_values('youtube_views', ascending=False)[:15]

alt.Chart(top_yt_views).mark_bar().encode(
    x = 'youtube_views',
    y = alt.Y('heroes', sort='-x'),
    color='category'
).properties(height=400, width=500).configure_view(
    stroke='transparent'
)
# youtube_likes
top_yt_like = episode.sort_values('youtube_likes', ascending=False)[:15]
alt.Chart(top_yt_like).mark_bar(color='green').encode(
    x = 'youtube_likes',
    y = alt.Y('heroes', sort='-x')
)
top_yt_dislike = episode.sort_values('youtube_dislikes', ascending=False)[:15]
alt.Chart(top_yt_dislike).mark_bar(color='firebrick').encode(
    x = 'youtube_dislikes',
    y = alt.Y('heroes', sort='-x')
)
top_yt_time = episode.sort_values('youtube_subscribers', ascending=False)[:15]
alt.Chart(top_yt_time).mark_bar().encode(
    x = 'youtube_subscribers',
    y = alt.Y('heroes', sort='-x')
)

top_yt_time = episode.sort_values('youtube_watch_hours', ascending=False)[:15]
alt.Chart(top_yt_time).mark_bar().encode(
    x = 'youtube_watch_hours',
    y = alt.Y('heroes', sort='-x')
)
episode['overall_likes'] = (episode['youtube_dislikes']+1) / episode['youtube_likes']
ctd_eps = episode[episode['episode_id'].str.match('E')]
ctd_eps = ctd_eps.drop([0])
top_yt_overall_like = ctd_eps.sort_values('overall_likes')[:15]
alt.Chart(top_yt_overall_like).mark_bar().encode(
    x = 'overall_likes',
    y = alt.Y('heroes', sort='-x')
)
episode['avg_duration'] = episode['youtube_watch_hours']*60/ episode['youtube_views']
top_yt_avg_time = episode.sort_values('avg_duration', ascending=False)[:15]
alt.Chart(top_yt_avg_time).mark_bar(color='green').encode(
    x = 'avg_duration',
    y = alt.Y('heroes',sort='-x')
)
time_dur = episode[['release_date', 'avg_duration', 'episode_duration']]
time_dur['episode_duration'] = time_dur['episode_duration'] / 60
time_dur = time_dur.set_index('release_date')
bcr.bar_chart_race(df = time_dur, title = "Average watch duration per episode in minutes", cmap='prism', figsize=(5, 3),)
fig = go.Figure(data=[go.Pie(labels=['Youtube organic views','Views from other sources'],
                             values=[episode['youtube_views'].sum()-episode['youtube_nonimpression_views'].sum(), episode['youtube_nonimpression_views'].sum(),], 
                            pull=[0.2, 0])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  )
fig.show()
# X-axis is the release_date
# y-axis can be the growth of subscribers
alt.Chart(ctd_eps).mark_line().encode(
    x = 'episode_id',
    y = alt.Y('youtube_subscribers')
)

ctd_eps['cum_youtube_views'] = ctd_eps['youtube_views'].cumsum()
fig = px.line(ctd_eps, x="episode_id", y="cum_youtube_views", title='Growth of youtube cumulative views across the episodes')
fig.show()
latest_thumbnail = episode.iloc[69:]
latest_thumbnail['youtube_ctr'].mean()
episode['youtube_ctr'].mean()
print("episode with max ctr", episode['youtube_ctr'].max())
episode[episode['youtube_ctr']==8.46]
tot_aud = ctd_eps['anchor_plays'].sum()
spotify_tot = ctd_eps['spotify_starts'].sum() # No of episode which was played for >0 s
apple_tot = ctd_eps['apple_listeners'].sum()
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=['Spotify','Apple Podcast','Others'],
                             values=[spotify_tot, apple_tot,tot_aud-(spotify_tot+apple_tot)])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
ctd_eps['spotify_starts'].sum()
ctd_eps['spotify_streams'].sum()
top_yt_duration = episode.sort_values('anchor_plays', ascending=False)[:15]
alt.Chart(top_yt_duration).mark_bar(color='black').encode(
    x = 'anchor_plays',
    y = alt.Y('heroes', sort='-x')
)
top_yt_duration = episode.sort_values('spotify_starts', ascending=False)[:15]
alt.Chart(top_yt_duration).mark_bar(color='green').encode(
    x = 'spotify_starts',
    y = alt.Y('heroes', sort='-x')
)
top_yt_duration = episode.sort_values('spotify_listeners', ascending=False)[:15]
alt.Chart(top_yt_duration).mark_bar(color='red').encode(
    x = 'spotify_listeners',
    y = alt.Y('heroes', sort='-x')
)
apple_listeners = episode['apple_listeners'].sum()
spotify_listeners = episode['spotify_listeners'].sum()

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=['Apple Listeners','Spotify Listeners'],
                             values=[apple_listeners, spotify_listeners])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
apple_avg_listeners = episode[['release_date', 'apple_avg_listen_duration', 'episode_duration']]
apple_avg_listeners = apple_avg_listeners.set_index('release_date')
bcr.bar_chart_race(df = apple_avg_listeners, title = "Average episode watch duration in Apple podcast(s)",  figsize=(5, 3),)