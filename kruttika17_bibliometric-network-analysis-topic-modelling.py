import os

import pandas as pd

import numpy as np

!pip install metaknowledge # ensure Internet setting is "On" 

import metaknowledge as mk

import networkx as nx

import community

import nltk

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import spacy

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import ldamodel

from gensim.models import CoherenceModel 

import re

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pprint import pprint

import warnings

warnings.filterwarnings("ignore")
# Importing the information science and bibliometrics dataset

RC = mk.RecordCollection("../input/mk/raw_data/imetrics/")

len(RC)
RC
# Saving the dataset as a csv file

RC.writeCSV("records.csv")

# Reading in the data as a Pandas dataframe

data = pd.read_csv("records.csv")

data.head(3)
# Saving the data as a dataframe using mk's makeDict()

data2 = pd.DataFrame(RC.makeDict())

data2.head(3)
# Printing basic statistics about the data

print(RC.glimpse())
# Generating the co-author network 

coauth_net = RC.networkCoAuthor()

coauth_net
# Printing the network stats

print(mk.graphStats(coauth_net))
mk.dropEdges(coauth_net, minWeight = 2, dropSelfLoops = True)

giant_coauth = max(nx.connected_component_subgraphs(coauth_net), key = len)

print(mk.graphStats(giant_coauth))
# Computing centrality scores

deg = nx.degree_centrality(giant_coauth)

clo = nx.closeness_centrality(giant_coauth)

bet = nx.betweenness_centrality(giant_coauth)

eig = nx.eigenvector_centrality(giant_coauth)



# Saving the scores as a dataframe

cent_df = pd.DataFrame.from_dict([deg, clo, bet, eig])

cent_df = pd.DataFrame.transpose(cent_df)

cent_df.columns = ["degree", "closeness", "betweenness", "eigenvector"]



# Printing the top 10 co-authors by degree centrality score

cent_df.sort_values("degree", ascending = False)[:10]
# Visualizing the top 10 co-authors by degree centrality score

sns.set(font_scale=.75)

cent_df_d10 = cent_df.sort_values('degree', ascending = False)[:10]

cent_df_d10.index.name = "author"

cent_df_d10.reset_index(inplace=True)

print()

plt.figure(figsize=(10,7))

ax = sns.barplot(y = "author", x = "degree", data = cent_df_d10, palette = "Set2");

ax.set_alpha(0.8)

ax.set_title("Top 10 authors in co-author graph", fontsize = 18)

ax.set_ylabel("Authors", fontsize=14);

ax.set_xlabel("Degree centrality", fontsize=14);

ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
solo = []

co = []

for i in cent_df_d10["author"]:

    # Calculate solo authorship rate

    so = np.round((len(data[(data['AF'].str.contains(i)) & (data["num-Authors"] == 1)])) / (len(data[data['AF'].str.contains(i)])), decimals = 2)

    solo.append(so)

    # Calculate co-authorship rate

    co.append(1-so)

print(solo, co)
# Create top 10 authors dataframe

authors = pd.DataFrame(zip(solo, co), columns = ["solo", "coauthor"])

authors["author"] = cent_df_d10["author"]

# Rearrange columns

authors = authors[["author", "solo", "coauthor"]]

authors
sns.set(rc={'figure.figsize':(8,7)})

fig = sns.swarmplot(x = "solo", y = "coauthor", hue = "author", data = authors, s = 12, palette = "Paired")

plt.title("Solo vs. Co-authorship rate: Top 10 Authors", fontsize = 16)

plt.xlabel("Solo authorship rate")

plt.ylabel("Co-authorship rate")

plt.show()
# Visualizing the co-author network

plt.figure(figsize = (10, 7))

size = [2000 * eig[node] for node in giant_coauth]

nx.draw_spring(giant_coauth, node_size = size, with_labels = True, font_size = 5,

               node_color = "#FFFFFF", edge_color = "#D4D5CE", alpha = .95)
# Community detection

partition = community.best_partition(giant_coauth) 

modularity = community.modularity(partition, giant_coauth)

print("Modularity:", modularity)
# Visualizing the communities

# Generates a different graph each time

plt.figure(figsize = (10, 7))

colors = [partition[n] for n in giant_coauth.nodes()]

my_colors = plt.cm.Set2 

nx.draw(giant_coauth, node_color=colors, cmap = my_colors, edge_color = "#D4D5CE")
# Transform the record collection into a format for use with natural language processing applications

data = RC.forNLP("topic_model.csv", lower=True, removeNumbers=True, removeNonWords=True, removeWhitespace=True)



# Convert the raw text into a list.

docs = data['abstract']

docs
# Defining a function to clean the text

def clean(docs):

    # Insert function for preprocessing the text

    def sent_to_words(sentences):

        for sentence in sentences:

            yield (simple_preprocess(str(sentence), deacc = True))

    # Tokenize the text

    tokens = sent_to_words(docs)

    # Create stopwords set

    stop = set(stopwords.words("english"))

    # Create lemmatizer

    lmtzr = WordNetLemmatizer()

    # Remove stopwords from text

    tokens_stopped = [[word for word in post if word not in stop] for post in tokens]

    # Lemmatize text

    tokens_cleaned = [[lmtzr.lemmatize(word) for word in post] for post in tokens_stopped]

    # Return cleaned text

    return tokens_cleaned



# Cleaning up the raw documents

cleaned_docs = clean(docs)

cleaned_docs
# Creating a dictionary

id2word = corpora.Dictionary(cleaned_docs)

print(id2word)
# Filtering infrequent and over frequent words

id2word.filter_extremes(no_below=5, no_above=0.5)

# Creating a document-term matrix

corpus = [id2word.doc2bow(doc) for doc in cleaned_docs]
# Building an LDA model with 5 topics

model = ldamodel.LdaModel(corpus = corpus, num_topics = 5, id2word = id2word, 

                              passes = 10, update_every = 1, chunksize = 1000, per_word_topics = True, random_state = 1)

# Printing the topic-word distributions

pprint(model.print_topics())
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds = "tsne")

vis