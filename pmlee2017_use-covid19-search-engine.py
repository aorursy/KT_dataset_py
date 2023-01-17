# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.layers import Input, Lambda, Dense

from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

from tqdm import tqdm_notebook



import helper_functions as hf

import meta_cleaning as mc

import eda_text as et

import config 

import viz_plot as vp

import word_cloud_prep as wcp

import covid_clustering as  cc



#import biobert_embedding as be

#import spacy

import matplotlib.pylab as plt

import plotly.express as px

from collections import defaultdict

from timeit import default_timer as timer

from IPython.display import Image

#import tabulate



#spacytokenizer = spacy.tokenizer.Tokenizer(be.nlp.vocab)



# Any results you write to the current directory are saved as output.

ROOTDIR = "../input"

DATADIR = os.path.join(ROOTDIR, 'CORD-19-research-challenge')
Image("../input/covid-image/cov.png")
df_meta = pd.read_csv(os.path.join(DATADIR, "metadata.csv"))
biorxiv_path = os.path.join(DATADIR, "biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/")

df_biorxiv = mc.parse_biorxiv(biorxiv_path)
comm_subset_path = os.path.join(

    DATADIR, "comm_use_subset/comm_use_subset/pdf_json/"

)

df_comm = mc.parse_comm(comm_subset_path)

noncomm_subset_path = os.path.join(

    DATADIR, "noncomm_use_subset/noncomm_use_subset/pdf_json/"

)

df_noncomm = mc.parse_noncomm(noncomm_subset_path)
df_meta.columns
# Convert publish time to type publish_date 

df_meta["publish_date"] =  pd.to_datetime(df_meta["publish_time"])

df_merge = mc.merge_datasets(df_meta, df_biorxiv, df_comm, df_noncomm)

df_merge_impute = mc.impute_columns(df_merge)

df_meta_comp = mc.drop_duplicates(df_merge_impute)

df_meta_comp.index.name ="row_id"

df_meta_comp.to_csv("df_meta_comp.csv")

del df_meta, df_merge_impute, df_comm, df_noncomm, df_biorxiv
df_meta_comp.head()

df_meta_comp.shape
mc.plot_missing_value_barchart(df_meta_comp)
df_authors = df_meta_comp["authors"].apply(mc.author_feats)

mc.plot_num_author_distrib(df_authors["num_authors"])

del df_authors
mc.plot_article_sources_distrib(df_meta_comp)
df_publish_date = mc.groupby_publish_date(df_meta_comp)

mc.plot_publish_date_distrib(df_publish_date)

del df_publish_date
df_date_source = mc.gropuby_date_source(df_meta_comp)

mc.plot_publish_date_wrt_sources(df_date_source)

del df_date_source
df_preprocess_title = df_meta_comp["title"].apply(

        lambda x: et.nlp_preprocess(x, config.PUNCT_DICT)

    )

    

df_wc_title = et.corpora_freq(df_preprocess_title)

et.plot_distrib(df_wc_title, "title");

del df_preprocess_title
df_preprocess_abstract = df_meta_comp["abstract"].apply(

        lambda x: et.nlp_preprocess(x, config.PUNCT_DICT)

    )

df_wc_abstract = et.corpora_freq(df_preprocess_abstract)

et.plot_distrib(df_wc_abstract, "abstract");



del df_preprocess_abstract
df_preprocess_text = df_meta_comp["text"].apply(

        lambda x: et.nlp_preprocess_text(x, config.PUNCT_DICT)

)

df_wc_text = et.corpora_freq(df_preprocess_text)

et.plot_distrib(df_wc_text, "text");

del df_preprocess_text
df_process_affiliation = df_meta_comp["affiliations"].apply(et.process_affiliations)



df_wc_affiliation = et.corpora_freq(df_process_affiliation, affiliation=True)

ax = et.plot_distrib(df_wc_affiliation.iloc[1:], "affiliation")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

del df_process_affiliation
Image("../input/semantic-similarity/Capture decran 2020-04-16 a 16.10.28.png")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

embed = hub.load(module_url)



def UniversalEmbedding(x):

    return embed(tf.squeeze(tf.cast(x, tf.string)))



input_text = Input(shape=(1,), dtype=tf.string)

embedding = Lambda(UniversalEmbedding, output_shape=(512, ))(input_text)

model = Model(inputs=[input_text], outputs=embedding)



# define some parameters

nsamples = 5000
title_len = df_meta_comp["title"].apply(lambda x: len(x.split()) if not pd.isnull(x) else np.NaN)

mean_title_len = np.nanmean(title_len)

print("Mean title length: %.1f words" %mean_title_len)
df_title = df_meta_comp["title"].dropna()



# keep id

id_titles = df_meta_comp[~df_meta_comp.title.isna()].index
embeddings_titles = model.predict(df_title, batch_size=1024, verbose=1)

df_embedding_title = pd.DataFrame(embeddings_titles, index = id_titles)

df_embedding_title.name = "row_id"

embeddings_titles = np.concatenate((id_titles.values.reshape(-1, 1), embeddings_titles), axis=1)

np.save('embeddings_titles.npy', embeddings_titles)

del embeddings_titles

n_clusters = np.arange(4, 32, 4)

df_clusters = cc.miniBatchClustering(df_embedding_title, None, n_clusters)
import matplotlib.pylab as plt

fig, axes = plt.subplots(3, 3, figsize=(12, 16))



sample_title = df_embedding_title.sample(nsamples, random_state=42)

idx_title = sample_title.index

cc.plot_silhouette_graph(

        df_embedding_title.sample(nsamples, random_state=42),

        df_clusters.loc[idx_title],

        n_clusters,

        fig,

        axes,

    )
df_calinski = cc.evaluate_metric_score(df_embedding_title, df_clusters, metric="calinski")

df_calinski["n_cluster"] = n_clusters
ax = cc.plot_metric_vs_cluster(df_calinski)

x = df_calinski['n_cluster'].values

y = df_calinski['metric'].values



a1 = (y[1]-y[0])/(x[1]-x[0])

b1 =1270



def f(x, a, b):

    return a*x+b



x_asympt= np.arange(4., 13.)

y1 = f(x_asympt, a1, b1)



ax.axhline(300,  color="black", linestyle="dashed", linewidth=1)

ax.axvline(11, ymax=.6, color="red", linestyle="dashed", linewidth=1)



ax.plot(x_asympt, y1, color="black", linestyle="dashed", linewidth=1)

ax.set_ylabel("Calinksi-Harabasz score");

X = vp.sphereize(df_embedding_title.values)

index = df_embedding_title.index

colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Plotly

df_pca = vp.compute_pca(X, index)

df_resampled = vp.resample(df_pca, df_meta_comp, df_clusters, colors, nsamples=nsamples, n_cluster=24)

info_title = vp.prepare_info(df_resampled)

vp.plot_tsne(df_resampled, "X_0", "X_1", info_title, var_z="X_2")
df_title_cluster = df_clusters.merge(

        df_title, left_index=True, right_index=True

    )
df_title_cluster["title_process"] = df_title_cluster["title"].apply(

        lambda x: et.nlp_preprocess(str(x), config.PUNCT_DICT)

        if not pd.isnull(x) else "")



df_title_wc = wcp.prepare_word_cloud(df_title_cluster["title_process"])

# get rid of top 10 words

extra_stopwords = df_title_wc.head(10).keys().tolist()



extra_stopwords += ["covid", "sars", "infectious", "19", "volume", "index", "chapter", "volume", "1", "de", "la"]

wc_title = defaultdict(int)

ncluster = 12

nclusters = sorted(

        df_title_cluster["labels_cluster_%d" % ncluster].unique().tolist()

    )



for k in nclusters:

    temp = df_title_cluster[df_title_cluster["labels_cluster_%d" % ncluster] == k][

        "title_process"

    ]



    try:

        wc_title[k] = wcp.prepare_word_cloud(temp, extra_stopwords)

    except ValueError:

        pass



n_top = 50

fig, axes = plt.subplots(4, 3, figsize=(16, 16))

ax = axes.flatten()

wcp.plot_word_cloud(wc_title, ax)

plt.tight_layout(w_pad=2.0)

del wc_title, df_title_cluster, df_title_wc
df_title_abstract = df_meta_comp['title'] + "\n" + df_meta_comp['abstract'].fillna('') 

df_title_abstract = df_title_abstract.dropna()

id_title_abstract = df_title_abstract[~df_title_abstract.isna()].index
title_abstract_len = df_title_abstract.apply(lambda x: len(x.split()) if not pd.isnull(x) else np.NaN)

mean_title_abstract_len = np.nanmean(title_abstract_len)

print("Mean title abstract length: %.1f words" %mean_title_abstract_len)
embeddings_title_abstract = model.predict(df_title_abstract, batch_size=1024, verbose=1)

df_embedding_title_abstract = pd.DataFrame(embeddings_title_abstract, index = id_title_abstract)

df_embedding_title_abstract.name = "row_id"

embeddings_title_abstract = np.concatenate((id_title_abstract.values.reshape(-1, 1), embeddings_title_abstract), axis=1)

np.save('embeddings_title_abstract.npy', embeddings_title_abstract)

del embeddings_title_abstract
n_clusters = np.arange(4, 30, 4)

df_clusters_title_abstract = cc.miniBatchClustering(df_embedding_title_abstract, None, n_clusters)
fig, axes = plt.subplots(3, 2, figsize=(12, 16))



sample_title_abstract = df_embedding_title_abstract.sample(nsamples, random_state=42)

idx_title_abstract = sample_title_abstract.index

cc.plot_silhouette_graph(

        df_embedding_title_abstract.sample(nsamples, random_state=42),

        df_clusters_title_abstract.loc[idx_title_abstract],

        n_clusters,

        fig,

        axes,

    )
df_calinski_title_abstract = cc.evaluate_metric_score(df_embedding_title_abstract, df_clusters_title_abstract, metric="calinski")

df_calinski_title_abstract["n_cluster"] = n_clusters


ax = cc.plot_metric_vs_cluster(df_calinski_title_abstract)

x = df_calinski_title_abstract['n_cluster'].values

y = df_calinski_title_abstract['metric'].values



a1 = (y[1]-y[0])/(x[1]-x[0])

b1 = 2350



def f(x, a, b):

    return a*x+b



x_asympt= np.arange(2., 14.)

y1 = f(x_asympt, a1, b1)



ax.axhline(500,  color="black", linestyle="dashed", linewidth=1)

ax.axvline(12, ymax=.5, color="red", linestyle="dashed", linewidth=1)

ax.plot(x_asympt, y1, color="black", linestyle="dashed", linewidth=1)



ax.set_ylabel("Calinski-Harabasz score");
X_title_abstract = vp.sphereize(df_embedding_title_abstract.values)

index_title_abstract = df_embedding_title_abstract.index

colors = px.colors.qualitative.Alphabet

df_pca_title_abstract = vp.compute_pca(X_title_abstract, index_title_abstract)

df_resampled_title_abstract = vp.resample(df_pca_title_abstract, df_meta_comp, 

                                          df_clusters_title_abstract, colors, nsamples=nsamples, n_cluster=12 )

info_title_abstract = vp.prepare_info(df_resampled_title_abstract)

vp.plot_tsne(df_resampled_title_abstract, "X_0", "X_1", info_title_abstract, var_z="X_2")
df_title_abstract.name = "title_abstract"

df_title_abstract_cluster = df_clusters_title_abstract.merge(

        df_title_abstract, left_index=True, right_index=True

    )

df_title_abstract_cluster["title_abstract_process"] = df_title_abstract_cluster["title_abstract"].apply(

        lambda x: et.nlp_preprocess(str(x), config.PUNCT_DICT)

        if not pd.isnull(x) else "")
df_title_abstract_wc = wcp.prepare_word_cloud(df_title_abstract_cluster["title_abstract_process"])

# since there are more words, we consider the top 20 words as stop words

extra_stopwords_title_abstract = df_title_abstract_wc.head(20).keys().tolist()



extra_stopwords_title_abstract += ["covid", "sars", "infectious", "19", "may", "can", "volume", "index", "chapter", "volume", "used", "also", "de", "la"]
wc_title_abstract = defaultdict(int)

ncluster = 12

nclusters = sorted(

        df_title_abstract_cluster["labels_cluster_%d" % ncluster].unique().tolist()

    )



for k in nclusters:

    temp = df_title_abstract_cluster[df_title_abstract_cluster["labels_cluster_%d" % ncluster] == k][

        "title_abstract_process"

    ]



    try:

        wc_title_abstract[k] = wcp.prepare_word_cloud(temp, extra_stopwords_title_abstract)

    except ValueError:

        pass



n_top = 50

fig, axes = plt.subplots(4, 3, figsize=(16, 16))

ax = axes.flatten()

wcp.plot_word_cloud(wc_title_abstract, ax)

plt.tight_layout(w_pad=2.0)

del wc_title_abstract, df_title_abstract_cluster, df_title_abstract_wc 
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

from ipywidgets import interact, interact_manual, Textarea, Layout
text_area_layout = Layout(width="70%", height="50px")

text_area = widgets.Textarea(value="Incubation periods", placeholder="Enter your text here.", layout=text_area_layout)



int_slider_layout = Layout(width="50%")

int_slider = widgets.IntSlider(description="Select number of results to show",

                               min=1, 

                               max=40, 

                               value=10, 

                               layout=int_slider_layout,

                               style={'description_width': 'initial'}

                              )



radio_buttons_layout = Layout(width="50%")

radio_buttons = widgets.RadioButtons(description="select embeddings", 

                                     value='title', 

                                     options=['title', 'title+abstract'],

                                     style={'description_width': 'initial'},

                                     layout=radio_buttons_layout

                                    )



toggle_button = widgets.ToggleButton(value=True)



checkbox = widgets.Checkbox(value=False, description='Show abstracts', disabled=False, indent=False)
@interact

def plot_search_results(emb=radio_buttons, n=int_slider, show_abstracts=checkbox, text=text_area):

    if text.strip() != '':

        if emb == "title": 

            embs = df_embedding_title

        elif emb == "title+abstract":

            embs = df_embedding_title_abstract

      

        print(f"Displaying {n} most similar results for \n{text} ...\n")

        

        embedding_text = embed([text])

        embedding_text = embedding_text.numpy()

        similarities = np.inner(embedding_text, embs.values)



        indices = np.argsort(similarities)[0]

        indices = indices[::-1][:n]



        row_ids = embs.iloc[indices].index

        row_ids = list(map(int, row_ids))



        for i, (row_id, index) in enumerate(zip(row_ids, indices)):



            title = df_meta_comp.loc[row_id]['title']

            abstract = df_meta_comp.loc[row_id]['abstract']

            print(f'result {i} title : {title}')

            print(f'similarity : {similarities[0][index]}')

            

            if show_abstracts:

                print('')

                print(f'result {i} abstract : {abstract}')



            print('----' )

    else:

        print('no query, no results baby.')
