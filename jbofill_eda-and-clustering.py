import numpy as np 

import pandas as pd 



import os



files_wanted = ['movies_metadata.csv']

file_paths = list()



for dirname, _, filenames in os.walk('/kaggle/input'):

     for filename in filenames:

            if filename in files_wanted:

                file_paths.append(str(dirname + "/" + filename))

meta_df = pd.read_csv(file_paths[0], low_memory=False)
meta_df.head()
meta_df.shape
meta_df.drop(['belongs_to_collection', 'homepage', 'tagline', 'poster_path', 'overview', 'imdb_id', 'spoken_languages'], inplace=True, axis=1)



column_changes = ['production_companies', 'production_countries', 'genres']



json_shrinker_dict = dict({'production_companies': list(), 'production_countries': list(), 'genres': list()})



meta_df.dropna(inplace=True)
import ast
for col in column_changes:

    if col == 'production_companies':

        for i in meta_df[col]:

            i = ast.literal_eval(i)

            if len(i) < 1:

                json_shrinker_dict['production_companies'].append(None)



            for element in i:

                json_shrinker_dict['production_companies'].append(element['name'])

                break

    elif col == 'production_countries':

        for i in meta_df[col]:

            i = ast.literal_eval(i)

            if len(i) < 1:

                json_shrinker_dict['production_countries'].append(None)

            for element in i:

                json_shrinker_dict['production_countries'].append(element['iso_3166_1'])

                break

    else:

        for i in meta_df[col]:

            i = ast.literal_eval(i)

            if len(i) < 1:

                json_shrinker_dict['genres'].append(None)



            for element in i:

                json_shrinker_dict['genres'].append(element['name'])

                break



for i in column_changes:

    meta_df[i] = json_shrinker_dict[i]



meta_df.dropna(inplace=True)



meta_df['budget'] = meta_df['budget'].astype(int)
meta_df.head()
meta_df.shape
import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)
fig = px.scatter(meta_df, x='budget', y='revenue', hover_data=['title'], color='genres', width=800, height=800)

fig.update_layout(

    title='The Relationship between Budget and Revenue',

    xaxis_title='Budget',

    yaxis_title='Revenue',

    font=dict(

        size=16

    )

)

iplot(fig)
genre_budget_df = meta_df.groupby(['genres'])['budget'].sum()



fig = go.Figure([

    go.Bar(

        x=genre_budget_df.index,

        y=genre_budget_df.values,

        text=genre_budget_df.values,

        textposition='auto',

        marker_color=['#94447f',

                      '#5796ef',

                      '#8a59c0',

                      '#288abf',

                      '#0ab78d',

                      '#4ed993',

                      '#7d3970',

                      '#b3dc67',

                      '#dc560a',

                      '#0079fe',

                      '#98d3a8',

                      '#d5105a',

                      '#d04dcf',

                      '#58c7a2',

                      '#7bf1f8',

                      '#244155',

                      '#587b77',

                      '#c64ac2',

                      '#5e805d',

                      '#ebab95']

    )])



fig.update_layout(

    title='Sum of all Movie Budgets in each Genre',

    xaxis_title='Genre',

    yaxis_title='Total Budget',

    width=800,

    height=1000,

    font=dict(

        size=16

    )

)



fig.layout.template = 'seaborn'



iplot(fig)
fig = px.scatter(meta_df, x='budget', y='runtime', hover_data=['title'], color='genres', width=800, height=800)

fig.update_layout(

    title='The Relationship between Budget and Movie Runtime',

    xaxis_title='Budget',

    yaxis_title='Runtime',

    font=dict(

        size=16

    )

)



iplot(fig)
fig = px.scatter(meta_df, y='runtime', x='revenue', hover_data=['title'], color='genres', width=800, height=800)

fig.update_layout(

    title='The Relationship between Runtime and Movie Revenue',

    yaxis_title='Runtime',

    xaxis_title='Revenue',

    font=dict(

        size=16

    )

)



iplot(fig)
fig = go.Figure(go.Box(

    y=meta_df['vote_count']

    

))



fig.update_layout(

    title='Vote Count Distribution',

    yaxis_title='Vote Count',

    width=800,

    height=800

)



iplot(fig)
meta_df = meta_df[meta_df['vote_count'] >= meta_df['vote_count'].quantile(.75)]

fig = go.Figure(go.Box(

    y=meta_df['vote_count']

))



fig.update_layout(

    title='Vote Count Distribution',

    yaxis_title='Vote Count',

    width=800,

    height=800

)



iplot(fig)
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()



scaled_df = meta_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']]



smaller_df = scaled_df.copy()



scaled = scalar.fit_transform(meta_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']])



scaled_df = pd.DataFrame(scaled, index=scaled_df.index, columns=scaled_df.columns)



scaled_df.head()
def apply_kmeans(df, clusters):

    kmeans = KMeans(n_clusters=clusters, random_state=0)

    cluster_labels = kmeans.fit(df).labels_

    string_labels = ["c{}".format(i) for i in cluster_labels]

    df['cluster_label'] = cluster_labels

    df['cluster_string'] = string_labels



    return df
def param_tune(df):

    scores = {'clusters': list(), 'score': list()}

    for cluster_num in range(1,31):

        scores['clusters'].append(cluster_num)

        scores['score'].append(KMeans(n_clusters=cluster_num, random_state=0).fit(df).score(df))



    scores_df = pd.DataFrame(scores)



    fig = go.Figure(go.Scatter(

        x=scores_df['clusters'],

        y=scores_df['score']

    ))



    fig.update_layout(

        xaxis_title='Cluster',

        yaxis_title='Score',

        title='Elbow Method Results',

        height=800,

        width=800

    )



    fig.show()



    return 9
clusters = param_tune(scaled_df)



scaled_df = apply_kmeans(scaled_df, clusters)
smaller_df = smaller_df.join(scaled_df[['cluster_label', 'cluster_string']])

smaller_df = smaller_df.join(meta_df[['title', 'genres']])



smaller_df.head()
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style
style.use('seaborn-poster')

fig, ax = plt.subplots(1,1)

cluster_comb = smaller_df.groupby(['cluster_label'])['title'].count()

sns.barplot(y=cluster_comb.index, x=cluster_comb.values, orient='h', palette="Spectral",

            edgecolor='black', linewidth=1)

plt.ylabel("Cluster", fontsize=18)

plt.xlabel("Records", fontsize=18)

plt.title("Records in Each Cluster", fontsize=20)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.show()
fig = px.scatter_matrix(smaller_df, dimensions=['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count'],

                        color='cluster_string', hover_data=['title', 'genres'])

fig.update_layout(

    title='Cluster Scatter Matrix',

    height=1000,

    width=800

)



iplot(fig)
clusters = list(smaller_df['cluster_label'].unique())

cluster_dict = dict()



cluster_count = 0

for col in range(3):

    for row in range(3):

        cluster_df = smaller_df[smaller_df.cluster_label == clusters[cluster_count]]

        cluster_dict["{},{}".format(str(col), str(row))] = cluster_df['genres'].value_counts()

        cluster_count += 1



cluster_count = 0



fig, axs = plt.subplots(3, 3, figsize=(15,15))



for col in range(3):

    for row in range(3):

        coord = "{},{}".format(str(col), str(row))



        sns.barplot(y=cluster_dict[coord].index, x=cluster_dict[coord].values, orient='h',

                    palette={'Drama': '#94447f',

                             'Action': '#5796ef',

                             'Adventure': '#8a59c0',

                             'Comedy': '#288abf',

                             'Crime': '#0ab78d',

                             'Thriller': '#4ed993',

                             'Fantasy': '#7d3970',

                             'Horror': '#b3dc67',

                             'Science Fiction': '#dc560a',

                             'Animation': '#0079fe',

                             'Romance': '#98d3a8',

                             'Mystery': '#d5105a',

                             'Family': '#d04dcf',

                             'War': '#58c7a2',

                             'History': '#7bf1f8',

                             'Western': '#244155',

                             'TV Movie': '#587b77',

                             'Music': '#c64ac2',

                             'Documentary': '#5e805d'}, edgecolor='black', linewidth=0.9, ax=axs[col][row])



        title = "Cluster {}'s Genre Distribution".format(cluster_count)

        axs[col][row].set_title(title, fontsize=15, fontweight='bold')

        cluster_count += 1

plt.tight_layout()

plt.show()
drama_df = smaller_df[(smaller_df.genres == 'Drama') & (smaller_df.cluster_label.isin([0, 1, 2, 5]))]

drama_df = drama_df.sort_values('cluster_label')



fig = px.violin(drama_df, y='revenue', x='cluster_string', color='cluster_string', points='all', hover_data=drama_df)



fig.update_layout(

    title='Revenue Distribution in Drama Movies',

    yaxis_title='Revenue',

    xaxis_title='Cluster',

    height=1000,

    width=800

)



iplot(fig)

fig = px.violin(drama_df, y='vote_average', x='cluster_string',

                    color='cluster_string', points='all', hover_data=drama_df)



fig.update_layout(

    title='Vote Average Distribution in Drama Movies',

    yaxis_title='Vote Average',

    xaxis_title='Cluster',

    height=1000,

    width=800

)



iplot(fig)
drama_df = smaller_df[(smaller_df.genres == 'Drama')]

drama_df = drama_df.sort_values('cluster_label')

fig = px.violin(drama_df, y='vote_average', x='cluster_string',

                color='cluster_string', points='all', hover_data=drama_df)



fig.update_layout(

    title='Vote Average Distribution in Drama Movies',

    yaxis_title='Vote Average',

    xaxis_title='Cluster',

    height=1000,

    width=800

)



iplot(fig)
drama_df = smaller_df

drama_df = drama_df.sort_values('cluster_label')

fig = px.violin(drama_df, y='vote_average', x='cluster_string',

                color='cluster_string', points='all', hover_data=drama_df)



fig.update_layout(

    title='Vote Average Distribution in all Movies',

    yaxis_title='Vote Average',

    xaxis_title='Cluster',

    height=1000,

    width=800

)



iplot(fig)