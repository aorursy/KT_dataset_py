#__________________
import json
import pandas as pd
#__________________
def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#____________________________
def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#_______________________________________
def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan
#_______________________________________
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews']
#_______________________________________
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users'}
#_______________________________________     
IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}
#_______________________________________
def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])
#_______________________________________
def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])
#_______________________________________
def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies
import matplotlib.pyplot as plt
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.graph_objs import *
import plotly.graph_objs as go
import numpy as np
import pandas as pd
#_______________________________________________
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
df = convert_to_original_format(movies, credits)
liste_genres = set()
for s in df['genres'].str.split('|'):
    liste_genres = set().union(s, liste_genres)
liste_genres = list(liste_genres)
liste_genres.remove('')
df_reduced = df[['actor_1_name', 'vote_average',
                 'title_year', 'movie_title']].reset_index(drop = True)
for genre in liste_genres:
    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)
df_reduced[:5]
df_actors = df_reduced.groupby('actor_1_name').mean()
df_actors.loc[:, 'favored_genre'] = df_actors[liste_genres].idxmax(axis = 1)
df_actors.drop(liste_genres, axis = 1, inplace = True)
df_actors = df_actors.reset_index()
df_actors[:10]
df_appearance = df_reduced[['actor_1_name', 'title_year']].groupby('actor_1_name').count()
df_appearance = df_appearance.reset_index(drop = True)
selection = df_appearance['title_year'] > 4
selection = selection.reset_index(drop = True)
most_prolific = df_actors[selection]
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(5, 5))
genre_count = []
for genre in liste_genres:
    genre_count.append([genre, df_reduced[genre].values.sum()])
genre_count.sort(key = lambda x:x[1], reverse = True)
labels, sizes = zip(*genre_count)
labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]
ax.pie(sizes, labels=labels_selected,
       autopct = lambda x:'{:2.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0)
ax.axis('equal')
plt.tight_layout()
reduced_genre_list = labels[:12]
trace=[]
for genre in reduced_genre_list:
    trace.append({'type':'scatter',
                  'mode':'markers',
                  'y':most_prolific.loc[most_prolific['favored_genre']==genre,'vote_average'],
                  'x':most_prolific.loc[most_prolific['favored_genre']==genre,'title_year'],
                  'name':genre,
                  'text': most_prolific.loc[most_prolific['favored_genre']==genre,'actor_1_name'],
                  'marker':{'size':10,'opacity':0.7,
                            'line':{'width':1.25,'color':'black'}}})
layout={'title':'Actors favored genres',
       'xaxis':{'title':'mean year of activity'},
       'yaxis':{'title':'mean score'}}
fig=Figure(data=trace,layout=layout)
pyo.iplot(fig)
selection = df_appearance['title_year'] > 10
most_prolific = df_actors[selection]
most_prolific
class Trace():
    #____________________
    def __init__(self, color):
        self.mode = 'markers'
        self.name = 'default'
        self.title = 'default title'
        self.marker = dict(color=color, size=110,
                           line=dict(color='white'), opacity=0.7)
        self.r = []
        self.t = []
    #______________________________
    def set_color(self, color):
        self.marker = dict(color = color, size=110,
                           line=dict(color='white'), opacity=0.7)
    #____________________________
    def set_name(self, name):
        self.name = name
    #____________________________
    def set_title(self, title):
        self.na = title
    #__________________________
    def set_values(self, r, t):
        self.r = np.array(r)
        self.t = np.array(t)
df2 = df_reduced[df_reduced['actor_1_name'] == 'Brad Pitt']
total_count  = 0
years = []
imdb_score = []
genre = []
titles = []
for s in liste_genres:
    icount = df2[s].sum()
    #__________________________________________________________________
    # Here, we set the limit to 3 because of a bug in plotly's package
    if icount > 3: 
        total_count += 1
        genre.append(s)
        years.append(list(df2[df2[s] == 1]['title_year']))
        imdb_score.append(list(df2[df2[s] == 1]['vote_average'])) 
        titles.append(list(df2[df2[s] == 1]['movie_title']))
max_y = max([max(s) for s in years])
min_y = min([min(s) for s in years])
year_range = max_y - min_y

years_normed = []
for i in range(total_count):
    years_normed.append( [360/total_count*((an-min_y)/year_range+i) for an in years[i]])
color = ['royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',
          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse']
trace = [Trace(color[i]) for i in range(total_count)]
tr    = []
for i in range(total_count):
    trace[i].set_name(genre[i])
    trace[i].set_title(titles[i])
    trace[i].set_values(np.array(imdb_score[i]),
                        np.array(years_normed[i]))
    tr.append(go.Scatter(r      = trace[i].r,
                         t      = trace[i].t,
                         mode   = trace[i].mode,
                         name   = trace[i].name,
                         marker = trace[i].marker,
#                         text   = ['default title' for j in range(len(trace[i].r))], 
                         hoverinfo = 'all'
                        ))        
layout = go.Layout(
    title='Brad Pitt movies',
    font=dict(
        size=15
    ),
    plot_bgcolor='rgb(223, 223, 223)',
    angularaxis=dict(        
        tickcolor='rgb(253,253,253)'
    ),
    hovermode='Closest',
)
fig = go.Figure(data = tr, layout=layout)
pyo.iplot(fig)
selection = df_appearance['title_year'] > 4
most_prolific = df_actors[selection]
actors_list = most_prolific['actor_1_name'].unique()
test = pd.crosstab(df['actor_1_name'], df['actor_2_name'])
edge = []
for actor_1, actor_2 in list(test[test > 0].stack().index):
    if actor_1 not in actors_list: continue
    if actor_2 not in actors_list: continue
   
    if actor_1 not in actors_list or actor_2 not in actors_list: continue
    if actor_1 != actor_2:
        edge.append([actor_1, actor_2])
num_of_adjacencies = [0 for _ in range(len(df_actors))]
for ind, col in df_actors.iterrows():
    actor = col['actor_1_name']
    nb = sum([1 for i,j in edge if (i == actor) or (j == actor)])
    num_of_adjacencies[ind] = nb
def prep(edge, num_of_adjacencies, df, actors_list):
    edge_trace = Scatter(
    x=[],
    y=[],
    line = Line(width=0.5,color='#888'),
    hoverinfo = 'none',
    mode = 'lines')
    
    for actor_1, actor_2 in edge:
        x0, y0 = df[df['actor_1_name'] == actor_1][['title_year', 'vote_average']].unstack()
        x1, y1 = df[df['actor_1_name'] == actor_2][['title_year', 'vote_average']].unstack()
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
             colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for ind, col in df.iterrows():
        if col['actor_1_name'] not in actors_list: continue
        node_trace['x'].append(col['title_year'])
        node_trace['y'].append(col['vote_average'])
        node_trace['text'].append(col['actor_1_name'])
        node_trace['marker']['color'].append(num_of_adjacencies[ind])
        
    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                    title='<br>Connections between actors',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=YAxis(showgrid=True, zeroline=False, showticklabels=True)))
    
    return fig
fig = prep(edge, num_of_adjacencies, df_actors, actors_list)
pyo.iplot(fig)