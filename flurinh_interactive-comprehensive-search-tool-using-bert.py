!pip install bert-extractive-summarizer
!pip install neuralcoref
!python -m spacy download en_core_web_md
!pip install -U sentence-transformers
import torch

print(torch.cuda.is_available())
import numpy as np

import pandas as pd

import json

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # print(os.path.join(dirname, filename))

        pass
from tqdm import tqdm
meta=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
titles = meta['title']

abstracts = meta['abstract']
print(meta.shape)

print(meta.keys())

print(titles[:10])

print(abstract[0])
def lower_cased(text):

    text = text.str.replace('[^a-zA-Z]', ' ', regex=True)

    text = text.str.lower()

    return text
def find_topic(df, topic='smoking', only_covid=False):

    if only_covid:

        print("Only gathering data concerning COVID-19.")

        filtered = df[(df['title'].str.contains(topic)) | (df['abstract'].str.contains(topic)) & (df['abstract'].str.contains['covid'] | df['abstract'].str.contains['sars'])]

    else:

        print("Extensive search (also including articles not mentioning COVID-19) ...")

        filtered = df[(df['title'].str.contains(topic)) | (df['abstract'].str.contains(topic))]

    print("Found {} matching articles for the topic \"{}\".".format(len(filtered), topic))

    return filtered
topic = 'smoking'

filtered = find_topic(meta, topic)
print(filtered[:10]['title'])
print(filtered[:3]['abstract'])
topic_list = ['Smoking', 

              'pre-existing pulmonary disease', 

              'pulmonary disease', 

              'Co-infections', 

              'respiratory infection', 

              'viral infection', 

              'transmissibility', 

              'virulency', 

              'Socio-economic factor', 

              'behavioral factor', 

              'economic impact']



for i, t in enumerate(topic_list):

    topic_list[i] = t.replace('[^a-zA-Z]', ' ').lower()



print(topic_list)
import spacy

import transformers

import neuralcoref

from tqdm import trange

from summarizer import Summarizer
model = Summarizer()
def lower_cased(text):

    if not isinstance(text, str):

        text = str(text)

    text = text.replace('[^a-zA-Z]', ' ')

    text = text.lower()

    return text
output = {}
def analysis(meta, topic_list, inspection_limit=500, text_body='abstract'):

    for topic in topic_list:

        summaries = []

        filtered = find_topic(meta, topic)

        for i in trange(len(filtered)):

            if i < inspection_limit:  # due to constraint of time we use a cutoff of 500 articles inspected

                if text_body is 'abstract':

                    text = filtered.iloc[i]['abstract']

                else:

                    # Todo: use full textbody

                    pass

                body = lower_cased(text).replace('background', '').replace('abstract', '').replace('introduction', '')

                result = model(body)

                result = ''.join(result)

                # print(result)

                article_id = filtered.iloc[i]['pubmed_id']

                summaries.append((result, article_id))

        output.update({topic: summaries})

    return output
output = analysis(meta, topic_list)
print(output['smoking'][0])
f = open("output.txt","w")

f.write(str(output))

f.close()
# for m in range(len(output['smoking'])):

#     print(output['smoking'][m][0])

#     print("")

#     print("")

#     print("")

#     print("")
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA





def cluster_embeddings(data, referenced_embeddings, pca=True, n_pcs=3):

    if pca:

        print("Preprocessing using PCA.")

        data = PCA(n_components=n_pcs).fit_transform(data)

    km = KMeans(init='k-means++')

    print("data shape", data.shape)

    clusters = km.fit_transform(data)

    labels = km.labels_

    centroids = km.cluster_centers_.tolist()

    referenced_clusters = []

    for i in range(len(referenced_embeddings)):

        referenced_clusters.append((data[i], referenced_embeddings[i][1]))

    return referenced_clusters, centroids, labels
def summary_clustering(output, topic_list, embedder, n_clusters=30, clustering_method='kmean', min_p=3, max_p=10):

    points = []

    data = None

    for t in topic_list:

        for i in range(len(output[t])):

            summary = output[t][i][0].split(".")

            n = len(summary)

            embedding = embedder.encode(summary)

            if clustering_method is 'kmean':

                if data is None:

                    data = np.vstack(embedding)

                else:

                    emb = np.vstack(embedding)

                    data = np.vstack([data, emb])

                for _ in range(n):

                    points.append((embedding, output[t][i][1]))

            else:

                if min_p > len(embedding):

                    raise NotImplemented

                elif max_p < len(embedding):

                    raise NotImplemented

                else:

                    for _ in range(n):

                        points.append((embedding, output[t][i][1]))

                    if data is None:

                        data = np.vstack(embedding)

                    else:

                        emb = np.vstack(embedding)

                        data = np.vstack([data, emb])

    print("Complete data table shape is {}.".format(data.shape))

    print("Number of points", len(points))

    return cluster_embeddings(data, points, n_clusters)
clusters, centroids, labels = summary_clustering(output, [topic_list[0]], embedder, clustering_method='kmean', min_p=3, max_p=10)
import plotly

import plotly.graph_objs as go

import random
def plot(clusters, centroids, labels):

    # Configure Plotly to be rendered inline in the notebook.

    plotly.offline.init_notebook_mode()



    centroids = np.asarray(centroids)

    for c in range(len(clusters)):

        if c == 0:

            data = clusters[c][0]

        else:

            data = np.vstack([data, clusters[c][0]])

    

    np.random.seed(12)

    colors = np.random.randn(len(centroids))

    labels = np.asarray(labels)

    color_dict = {}

    for _ in range(colors.shape[0]):

        color_dict.update({_:colors[_]})

    color_points = []

    for l in range(len(labels)):

        label = labels[l]

        color_points.append(color_dict[label])

    colors_p = np.asarray(color_points)

    

    # Configure the trace.

    trace = go.Scatter3d(

        x=centroids[:,0],

        y=centroids[:,1],

        z=centroids[:,2],

        mode='markers',

        marker={

            'size': 10,

            'opacity': 0.8,

            'color': colors

        }

    )



    # Configure the trace.

    trace_2 = go.Scatter3d(

        x=data[:,0],

        y=data[:,1],

        z=data[:,2],

        mode='markers',

        marker={

            'size': 4,

            'opacity': 0.4,

            'color': colors_p

        }

    )

    



    # Configure the layout.

    layout = go.Layout(

        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}

    )



    data = [trace, trace_2]



    fig = go.Figure(data=data, layout=layout)

    fig.layout.title = 'Torque and Fuel Efficience'

    # Render the plot.

    plotly.offline.iplot(fig)
plot(clusters, centroids, labels)