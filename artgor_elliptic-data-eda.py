!pip install -U vega_datasets notebook vega
import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

pd.options.display.precision = 15



import time

import datetime

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from IPython.display import HTML

import json

import altair as alt



import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline



alt.renderers.enable('notebook')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import altair as alt

from altair.vega import v5

from IPython.display import HTML



# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

def prepare_altair():

    """

    Helper function to prepare altair for working.

    """



    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

    noext = "?noext"

    

    paths = {

        'vega': vega_url + noext,

        'vega-lib': vega_lib_url + noext,

        'vega-lite': vega_lite_url + noext,

        'vega-embed': vega_embed_url + noext

    }

    

    workaround = f"""    requirejs.config({{

        baseUrl: 'https://cdn.jsdelivr.net/npm/',

        paths: {paths}

    }});

    """

    

    return workaround

    



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

           



@add_autoincrement

def render(chart, id="vega-chart"):

    """

    Helper function to plot altair visualizations.

    """

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )
df_classes = pd.read_csv('/kaggle/input/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')

df_features = pd.read_csv('/kaggle/input/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

df_edgelist = pd.read_csv('/kaggle/input/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
df_classes['class'].value_counts()
df_features.head()
# renaming columns

df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
df_features['time step'].value_counts().sort_index().plot();

plt.title('Number of transactions in each time step');
# merge with classes

df_features = pd.merge(df_features, df_classes, left_on='id', right_on='txId', how='left')
plt.figure(figsize=(12, 8))

grouped = df_features.groupby(['time step', 'class'])['id'].count().reset_index().rename(columns={'id': 'count'})

sns.lineplot(x='time step', y='count', hue='class', data=grouped);

plt.legend(loc=(1.0, 0.8));

plt.title('Number of transactions in each time step by class');
bad_ids = df_features.loc[(df_features['time step'] == 37) & (df_features['class'] == '1'), 'id']

short_edges = df_edgelist.loc[df_edgelist['txId1'].isin(bad_ids)]
graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 

                                 create_using = nx.DiGraph())

pos = nx.spring_layout(graph)

nx.draw(graph, cmap = plt.get_cmap('rainbow'), with_labels=True, pos=pos)
graph1 = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 

                                 create_using = nx.Graph())

pos1 = nx.spring_layout(graph1)

nx.draw(graph1, cmap = plt.get_cmap('rainbow'), with_labels=False, pos=pos1)
# grouped = df_features.groupby(['time step', 'class'])['trans_feat_0'].mean().reset_index()

# chart = alt.Chart(grouped).mark_line().encode(

#     x=alt.X("time step:N", axis=alt.Axis(title='Time step', labelAngle=315)),

#     y=alt.Y('trans_feat_0:Q', axis=alt.Axis(title='Mean of trans_feat_0')),

#     color = 'class:N',

#     tooltip=['time step:O', 'trans_feat_0:Q', 'class:N']

# ).properties(title="Average trans_feat_0 in each time step by type", width=600).interactive()

# chart
plt.figure(figsize=(12, 8))

grouped = df_features.groupby(['time step', 'class'])['trans_feat_0'].mean().reset_index()

sns.lineplot(x='time step', y='trans_feat_0', hue='class', data=grouped);

plt.legend(loc=(1.0, 0.8));

plt.title('Average trans_feat_0 in each time step by type');