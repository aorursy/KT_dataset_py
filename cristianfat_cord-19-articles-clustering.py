import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

df[:3]
df[:3]
df = df[['title','abstract','authors']]
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df[df.duplicated()].__len__()
df.drop_duplicates(inplace=True)

df[df.duplicated()].__len__()
df[:3]
import tensorflow as tf

import tensorflow_hub as hub

class UniversalSenteneceEncoder:



    def __init__(self, encoder='universal-sentence-encoder', version='4'):

        self.version = version

        self.encoder = encoder

        self.embd = hub.load(f"https://tfhub.dev/google/{encoder}/{version}")



    def embed(self, sentences):

        return self.embd(sentences)



    def squized(self, sentences):

        return np.array(self.embd(tf.squeeze(tf.cast(sentences, tf.string))))
encoder = UniversalSenteneceEncoder(encoder='universal-sentence-encoder',version='4')
%%time

df = df[:5000]

df['title_sent_vects'] = encoder.squized(df.title.values).tolist()

# df['abstract_sent_vects'] = encoder.embed(df.abstract.values)
import plotly.graph_objects as go



sents = 50

labels = df[:sents].title.values

features = df[:sents].title_sent_vects.values.tolist()



fig = go.Figure(data=go.Heatmap(

                    z=np.inner(features, features),

                    x=labels,

                    y=labels,

                    colorscale='Viridis',

                    ))



fig.update_layout(

    margin=dict(l=40, r=40, t=40, b=40),

    height=1000,

    xaxis=dict(

        autorange=True,

        showgrid=False,

        ticks='',

        showticklabels=False

    ),

    yaxis=dict(

        autorange=True,

        showgrid=False,

        ticks='',

        showticklabels=False

    )

)



fig.show()
%%time

df = df[:5000]

df['abstract_sent_vects'] = encoder.squized(df.abstract.values).tolist()

# df['abstract_sent_vects'] = encoder.embed(df.abstract.values)
import plotly.graph_objects as go



sents = 30

labels = df[:sents].abstract.values

features = df[:sents].abstract_sent_vects.values.tolist()



fig = go.Figure(data=go.Heatmap(

                    z=np.inner(features, features),

                    x=labels,

                    y=labels,

                    colorscale='Viridis',

                    ))



fig.update_layout(

    margin=dict(l=140, r=140, t=140, b=140),

    height=1000,

    xaxis=dict(

        autorange=True,

        showgrid=False,

        ticks='',

        showticklabels=False

    ),

    yaxis=dict(

        autorange=True,

        showgrid=False,

        ticks='',

        showticklabels=False

    ),

    hoverlabel = dict(namelength = -1)

)



fig.show()
from sklearn.cluster import KMeans

import matplotlib.cm as cm

from sklearn.decomposition import PCA

from datetime import datetime

import plotly.express as px
%%time

n_clusters = 10

vectors = df.title_sent_vects.values.tolist()

kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 0)

kmean_indices = kmeans.fit_predict(vectors)



pca = PCA(n_components=512)

scatter_plot_points = pca.fit_transform(vectors)



tmp = pd.DataFrame({

    'Feature space for the 1st feature': scatter_plot_points[:,0],

    'Feature space for the 2nd feature': scatter_plot_points[:,1],

    'labels': kmean_indices,

    'title': df.title.values.tolist()[:vectors.__len__()]

})



fig = px.scatter(tmp, x='Feature space for the 1st feature', y='Feature space for the 2nd feature', color='labels',

                 size='labels', hover_data=['title'])

fig.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),

    height=1000

)



fig.show()
%%time

n_clusters = 10

vectors = df.abstract_sent_vects.values.tolist()

kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 0)

kmean_indices = kmeans.fit_predict(vectors)



pca = PCA(n_components=512)

scatter_plot_points = pca.fit_transform(vectors)



tmp = pd.DataFrame({

    'Feature space for the 1st feature': scatter_plot_points[:,0],

    'Feature space for the 2nd feature': scatter_plot_points[:,1],

    'labels': kmean_indices,

    'title': df.abstract.values.tolist()[:vectors.__len__()]

})



fig = px.scatter(tmp, x='Feature space for the 1st feature', y='Feature space for the 2nd feature', color='labels',

                 size='labels', hover_data=['title'])

fig.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),

    height=1000

)



fig.show()
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
%%time 

sdf = df.copy()

similarity_matrix = cosine_similarity(sdf.title_sent_vects.values.tolist())
simdf = pd.DataFrame(

    similarity_matrix,

    columns = sdf.title.values.tolist(),

    index = sdf.title.values.tolist()

)
long_form = simdf.unstack()

# rename columns and turn into a dataframe

long_form.index.rename(['t1', 't2'], inplace=True)

long_form = long_form.to_frame('sim').reset_index()
long_form = long_form[long_form.t1 != long_form.t2]

long_form[:3]
!pip install python-igraph
import igraph as ig
%%time

lng = long_form[long_form.sim > 0.75] 

tuples = [tuple(x) for x in lng.values]

Gm = ig.Graph.TupleList(tuples, edge_attrs = ['sim'])

layt=Gm.layout('kk', dim=3)
Xn=[layt[k][0] for k in range(layt.__len__())]# x-coordinates of nodes

Yn=[layt[k][1] for k in range(layt.__len__())]# y-coordinates

Zn=[layt[k][2] for k in range(layt.__len__())]# z-coordinates

Xe=[]

Ye=[]

Ze=[]

for e in Gm.get_edgelist():

    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends

    Ye+=[layt[e[0]][1],layt[e[1]][1], None]

    Ze+=[layt[e[0]][2],layt[e[1]][2], None]
import plotly.graph_objs as go
trace1 = go.Scatter3d(x = Xe,

  y = Ye,

  z = Ze,

  mode = 'lines',

  line = dict(color = 'rgb(0,0,0)', width = 1),

  hoverinfo = 'none'

)



trace2 = go.Scatter3d(x = Xn,

  y = Yn,

  z = Zn,

  mode = 'markers',

  name = 'articles',

  marker = dict(symbol = 'circle',

    size = 6, 

    # color = group,

    colorscale = 'Viridis',

    line = dict(color = 'rgb(50,50,50)', width = 0.5)

  ),

  text = lng.t1.values.tolist(),

  hoverinfo = 'text'

)



axis = dict(showbackground = False,

  showline = False,

  zeroline = False,

  showgrid = False,

  showticklabels = False,

  title = ''

)



layout = go.Layout(

  title = "Network of similarity between CORD-19 Articles(3D visualization)",

  width = 1500,

  height = 1500,

  showlegend = False,

  scene = dict(

    xaxis = dict(axis),

    yaxis = dict(axis),

    zaxis = dict(axis),

  ),

  margin = dict(

    t = 100,

    l = 20,

    r = 20

  ),



)
fig=go.Figure(data=[trace1,trace2], layout=layout)



fig.show()
sim_weight = 0.75

gdf = long_form[long_form.sim > sim_weight]
plt.figure(figsize=(50,50))

pd_graph = nx.Graph()

pd_graph = nx.from_pandas_edgelist(gdf, 't1', 't2')

pos = nx.spring_layout(pd_graph)

nx.draw_networkx(pd_graph,pos,with_labels=True,font_size=10, node_size = 30)
betCent = nx.betweenness_centrality(pd_graph, normalized=True, endpoints=True)

node_color = [20000.0 * pd_graph.degree(v) for v in pd_graph]

node_size =  [v * 10000 for v in betCent.values()]

plt.figure(figsize=(35,35))

nx.draw_networkx(pd_graph, pos=pos, with_labels=True,

                 font_size=5,

                 node_color=node_color,

                 node_size=node_size )
l=list(nx.connected_components(pd_graph))



L=[dict.fromkeys(y,x) for x, y in enumerate(l)]



d=[{'articles':k , 'groups':v }for d in L for k, v in d.items()]
gcd = pd.DataFrame.from_dict(d)
import nltk

nltk.download('stopwords')

nltk.download('punkt') 

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.tokenize import RegexpTokenizer



tok = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english')) 

def clean(string):

    return " ".join([w for w in word_tokenize(" ".join(tok.tokenize(string))) if not w in stop_words])



gcd.articles = gcd.articles.apply(lambda x: clean(x))
gcd.__len__(),gcd.__len__() / df.__len__()
from wordcloud import WordCloud
%%time

clouds = dict()



big_groups = pd.DataFrame({

    'counts':gcd.groups.value_counts()

    }).sort_values(by='counts',ascending=False)[:9].index.values.tolist()



for group in big_groups:

    text = gcd[gcd.groups == group].articles.values

    wordcloud = WordCloud(width=1000, height=1000).generate(str(text))

    clouds[group] = wordcloud
def plot_figures(figures, nrows = 1, ncols=1):

    """Plot a dictionary of figures.



    Parameters

    ----------

    figures : <title, figure> dictionary

    ncols : number of columns of subplots wanted in the display

    nrows : number of rows of subplots wanted in the figure

    """



    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=(20,20))

    for ind,title in zip(range(len(figures)), figures):

        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())

        axeslist.ravel()[ind].set_title(f'Most Freqent words for the group {title+1}')

        axeslist.ravel()[ind].set_axis_off()

    plt.tight_layout() # optional
plot_figures(clouds, 3, 3)

plt.show()