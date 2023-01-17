import pandas as pd

import yake_helper_funcs as yhf

import numpy as np

from nltk.tokenize import RegexpTokenizer

from datetime import datetime

forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")



forum_posts["Message"] = forum_posts["Message"].astype(str)

forum_posts["PostDate"] = pd.to_datetime(forum_posts["PostDate"])
tokenizer = RegexpTokenizer(r'\w+')

forum_posts["Message"] = [w.lower() for w in forum_posts["Message"].tolist()]

forum_posts["Message"] = [tokenizer.tokenize(i) for i in forum_posts["Message"]]
months_lookback = 12
today = datetime.today()
from dateutil.relativedelta import relativedelta

month_ranges = [pd.to_datetime(today)]

for months in range(months_lookback):

    month_ranges.append(pd.to_datetime(today - relativedelta(months=months + 1)))
dict_of_dfs = {}

month_shapes = []

for i in range(len(month_ranges) - 1):

    date_range_df = forum_posts.loc[(forum_posts['PostDate'] < month_ranges[i]) & (forum_posts['PostDate'] > month_ranges[i + 1])]

    month_shapes.append(date_range_df.shape[0])

    dict_of_dfs[month_ranges[i]] = date_range_df
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format("../input/fine-tuning-word2vec-2-0/kaggle_word2vec.model", binary = False)
def vectors_from_post(posts):

    post_vectors = np.zeros(shape = (len(posts), 300))

    for i, post in enumerate(posts):

        try:

            post_vectors[i] = w2v[post].mean(axis = 0)

        except:

            #text is empty no vector added

            pass

    return post_vectors
months_vectors = []
for values in dict_of_dfs.values():

    months_vectors.append(vectors_from_post(values["Message"].tolist()))
all_vecs = np.concatenate(months_vectors, axis = 0)
from plotly import offline

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from sklearn.decomposition import TruncatedSVD

n_iters = 100



reducer = TruncatedSVD(n_components=3, n_iter = 100)

reduced_dimensions = reducer.fit_transform(all_vecs)



    

def add_plot(reduced_dimensions, text, name, color = 'rgba(255, 255, 0, .5)', words_to_show = -1):

    init_notebook_mode(connected=True)

    print(len(reduced_dimensions[:words_to_show,0]))

    print(len(text[:words_to_show]))

    embeds = go.Scatter3d(

        name = name,

        x=reduced_dimensions[:words_to_show,0],

        y=reduced_dimensions[:words_to_show,1],

        z=reduced_dimensions[:words_to_show,2],

        mode='markers',

        text = text[:words_to_show],

        marker=dict(

            size=12,

            line=dict(

                color=color,

                width=0.1

            ),

            opacity=1.0

        )

    )

    return embeds



month_indexes = 0

words_to_show = 100

data = []

for i, (key, value) in enumerate(dict_of_dfs.items()):

    text = [" ".join(sent) for sent in value["Message"]]

    vecs = all_vecs[month_indexes:month_indexes + month_shapes[i], :]

    month_indexes += month_shapes[i]

    data.append(add_plot(vecs, text, name = str(key), words_to_show = words_to_show))

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple-3d-scatter')