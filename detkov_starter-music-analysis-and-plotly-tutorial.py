import numpy as np

import pandas as pd



from ast import literal_eval

from collections import Counter

import re



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
df = pd.read_csv('../input/lyrics-dataset/songs_dataset.csv')

df.head()
print(f'Dataset contains {df.shape[0]} songs')
df['Featuring'] = df['Featuring'].apply(literal_eval)

df['Genre'] = df['Genre'].apply(literal_eval)

df['Tags'] = df['Tags'].apply(literal_eval)

df['Producers'] = df['Producers'].apply(literal_eval)

df['Writers'] = df['Writers'].apply(literal_eval)

df['Date'] = pd.to_datetime(df['Date'])
null_cols = df.isnull().sum()

print('Number of missing values:')

print(null_cols[null_cols != 0])

print('Number of missing values in pct:')

print((null_cols[null_cols != 0] / df.shape[0]).apply(lambda x: f'{x*100:.2f}%'))
df = df[pd.isnull(df['Date']) | (df['Date'] >= '1950-01-01')].reset_index(drop=True)
df[df['Date'] >= '2020-01-01'].head()
df = df[df['Date'] < '2020-01-01']
df['Year'] = df['Date'].apply(lambda x: x.year)

# yes, you can put here any aggregation function you want, not only 'max', 'mean', etc

interval_aggregation = df.groupby(['Year'], as_index=False).agg({'Singer': pd.Series.nunique,

                                                                 'Song': pd.Series.count,

                                                                 'Album': pd.Series.nunique})

interval_aggregation.head()
# https://plot.ly/python/bar-charts/

fig = px.bar(interval_aggregation, x='Year', y='Song',

             # columns values that can be observed while hover over bar

             hover_data=['Singer', 'Album'],

             # just an alias to column names to be shown on chart

             labels={'Year': 'Year',

                     'Song': 'Number of songs', 

                     'Singer': 'Number of singers',

                     'Album': 'Number of albums'},

             # color of the bar will depend on value of this column

             color='Song',

             # you can play with different colormaps: 

             # https://plot.ly/python/colorscales/ 

             # https://plot.ly/python/plotly-express/

             color_continuous_scale='Cividis',

             # you can specify a title, its adjustment will be shown further

             title='Distribution of number of songs over years'   

            )

fig.show()
# you can put here any N you want, in my case N=3. 

# works right out of the box.

years_in_interval = 3



min_year = df['Date'].min().year

max_year = df["Date"].max().year

""

start = min_year - 1

end = max_year + years_in_interval - ((max_year - min_year) % years_in_interval)



year_bins = pd.date_range(f'{start}-12-31', f'{end}-01-01', freq=f'{years_in_interval}Y')

year_bins_labels = [f'{year_bins[i-1].year+1}-{year_bins[i].year}' 

                    for i in range(1, len(year_bins))]
df['Date_bin'] = pd.cut(df['Date'], 

                        bins=year_bins, 

                        labels=year_bins_labels)

df.head()
interval_aggregation = df.groupby(['Date_bin'], as_index=False).agg({'Singer': pd.Series.nunique, 

                                                                     'Song': pd.Series.count,

                                                                     'Album': pd.Series.nunique})

# we will be original and remember about ".tail()"

interval_aggregation.tail()
fig = px.bar(interval_aggregation, x='Date_bin', y='Song',

             hover_data=['Singer', 'Album'],

             labels={'Date_bin': 'Time interval',

                     'Song': 'Number of songs',

                     'Singer': 'Number of singers',

                     'Album': 'Number of albums'},

             color='Song',

             color_continuous_scale='Magma'

            )



x_tick_vals = list(range(0, interval_aggregation['Song'].max() + 2000, 2000))



# https://plot.ly/python/tick-formatting/ - formatting ticks

# https://plot.ly/python/axes/ - setting axes

# https://plot.ly/python/figure-labels/ - setting font

fig.update_layout(

    title=dict(text='Distribution of number of songs over year bins', 

               font_size=30,

               font_family='Ubuntu, monospace',

               font_color='#C23C74'

    ),

    xaxis=dict(

        showgrid=True,

        ticks='outside', 

        ticklen=10,

        tickcolor='#FA8965',

        color='black'

    ),

    yaxis=dict(

        showgrid=True,

        tickvals=x_tick_vals,

        ticktext=[' ']+[f'{x//1000}k' for x in x_tick_vals][1:],

        color='#390F68'

    )

)



fig.show()
genre_dict = Counter([tag for tags in df['Genre'].tolist() for tag in tags])

genre_dict.most_common()
genres = {'Hip-Hop/Rap': None, 'Rock': None}

n_singers = 10



for genre in genres:

    # check if genre types are in tags

    df[f'is_{genre}'] = df['Genre'].apply(lambda x: not bool(set(x).isdisjoint([genre]))) 

    

    top_singers = (df[df[f'is_{genre}']]

                   .groupby('Singer', as_index=False)

                   ['Song'].count()

                   .sort_values(by='Song', ascending=False)

                   .iloc[:n_singers, 0])

    singer_songs = (df[df['Singer'].isin(top_singers)]

                    .groupby(['Singer', 'Year'], as_index=False)

                    .agg({'Song': pd.Series.count,

                          'Album': pd.Series.nunique}))

    

    singer_songs['Song'] = singer_songs.groupby(['Singer'])['Song'].cumsum()

    singer_songs['Album'] = singer_songs.groupby(['Singer'])['Album'].cumsum()

    genres[genre] = singer_songs
# black magic here, be careful!

figs = []

for genre in genres:

    figs.append(px.scatter(genres[genre], x="Year", y="Song", 

                           color='Singer', 

                           size='Album', 

                           hover_name='Singer',

                           labels={'Year': 'Year',

                                   'Song': 'Number of songs',

                                   'Album': 'Number of albums'}

                          ))



# it's a little bit of illegal to do subplots with  px

# but we are not that simple: we can hardcode this!

# https://github.com/plotly/plotly_express/issues/83

traces = [fig['data'] for fig in figs]



# https://plot.ly/python/subplots/

fig = make_subplots(rows=len(genres), cols=1, 

                    vertical_spacing=0.05,

                    shared_xaxes=True,

                    subplot_titles=list(genres.keys()))



for i in range(len(genres)):

    for trace in traces[i]:

        fig.add_trace(trace, row=i+1, col=1)



fig.update_layout(title=dict(text='Number of songs of singers over the years', 

                             font_size=30,

                             font_family='Ubuntu'

                            ),

                  height=400*len(genres), 

                  showlegend=False,

                  font_size=16,

                  font_family='Ubuntu'

                 )

# annotations here are actually subplots titles 

for i in fig['layout']['annotations']:

    i['font'] = dict(size=20, family='Ubuntu')



# to show not only markers, but also connecting them with lines

fig.update_traces(mode='lines+markers')

fig.show()
df_hip_hop = df[df['is_Hip-Hop/Rap']].copy()

# hip-hop before 90s differs a lot from its current state  

df_hip_hop = df_hip_hop[df_hip_hop['Year'] >= 1990]
# for better tokenization 

def format_lyrics(lyrics):

    lyrics = re.sub('[*.,!:?\"\'«»]', '', lyrics)

    lyrics = re.sub('[-–—— ]+', ' ', lyrics)

    lyrics = lyrics.strip()

    lyrics = lyrics.lower()

    return lyrics
df_hip_hop['Lyrics'] = df_hip_hop['Lyrics'].apply(format_lyrics)

words = df_hip_hop['Lyrics'].apply(lambda x: x.split())



df_hip_hop['number_of_words'] = words.apply(len)

df_hip_hop['unique_words_count'] = words.apply(lambda x: len(list(set(x))))

df_hip_hop['unique_words_proportion'] = df_hip_hop['unique_words_count'] / df_hip_hop['number_of_words']

df_hip_hop.head()
unique_words_df = df_hip_hop.groupby(['Year'], as_index=False).agg({'Song': pd.Series.count, 

                                                                    'number_of_words': 'mean',

                                                                    'unique_words_count': 'mean',

                                                                    'unique_words_proportion': 'mean'})

unique_words_df.head()
# this way you can do second axis

# https://plot.ly/python/multiple-axes/

fig = make_subplots(specs=[[{"secondary_y": True}]])



# you shold add each plot to figure separately to left and right Y axis

fig.add_trace(

    go.Scatter(x=unique_words_df['Year'], 

               y=unique_words_df['number_of_words'], 

               mode='lines+markers',

               text=unique_words_df[['Year', 'Song']],

               name='All words',

              ), 

    secondary_y=False

)

fig.add_trace(

    go.Scatter(x=unique_words_df['Year'], 

               y=unique_words_df['unique_words_count'], 

               mode='lines+markers',

               text=unique_words_df['Year'],

               name='Unique words',

              ), 

    secondary_y=False

)

fig.add_trace(

    go.Scatter(x=unique_words_df['Year'], 

               y=unique_words_df['unique_words_proportion'], 

               mode='lines+markers',

               text=unique_words_df['Year'],

               name='Unique words proportion',

               marker_color='green' # paint right Y axis with one color

              ), 

    secondary_y=True

)



fig.update_layout(

    title=dict(text='How Hip-Hop lyrics are changing over years', 

               font_size=30,

               font_family='Ubuntu'

              ),

)



# this way you can work with different axis (left and right)

fig.update_yaxes(title_text='Mean <b>Number</b> of words', 

                 secondary_y=False)

fig.update_yaxes(title_text='Mean <b>Proportion</b> of words', 

                 color='green',  # paint right Y axis with one color

                 showgrid=False, # to remove additional ugly grid

                 secondary_y=True)



fig.show()
singer = 'Eminem'

temp = df[df['Singer'] == singer]

singer_feats_count = Counter([tag for tags in temp['Featuring'].tolist() for tag in tags])

singer_feats_count = Counter({key: val for key, val in singer_feats_count.items() if val > 1})
fig = go.Figure(data=[go.Pie(labels=list(singer_feats_count.keys()), 

                             values=list(singer_feats_count.values()), 

                             hole=0.2

                            )])



fig.update_traces(hoverinfo='label+value',

                  textinfo='value',

                  textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=1))

                 )



fig.update(layout_title_text=f'{singer}\'s feats',

           layout_showlegend=False

          )



fig.show()
