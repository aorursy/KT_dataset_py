# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, re, warnings

warnings.filterwarnings("ignore")



# Plot

import matplotlib.pyplot as plt

import seaborn as sns
# Loading Universal Sentence Encoder



import tensorflow as tf

import tensorflow_hub as hub



encoder_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'     # Note that this is 1GB Module



encoder = hub.load(encoder_url)
# Custom Function

def embed(input):

    return encoder(input)
word = "Tutorial"

sentence = "The input is variable length English text and the output is a 512 dimensional vector"

paragraph = (

    "The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, "

    "semantic similarity, clustering and other natural language tasks. The model is trained and optimized for greater-than-word length text, "

    "such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically "

    "accommodating a wide variety of natural language understanding tasks")



messages = [word, sentence, paragraph]

message_embeddings = embed(messages)



for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):

    print("Message: {}".format(messages[i]))

    print("Embedding size: {}".format(len(message_embedding)))

    message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))

    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
# Custom Function to Estimate & then plot(heatmap) the text similarity



def plot_similarity(labels, features, rotation):

    corr = np.inner(features, features)

    sns.set(font_scale=1.2)

    plt.figure(figsize=(10,10))

    g = sns.heatmap(corr, xticklabels=labels, yticklabels=labels,

                    vmin=0, vmax=1, cmap='coolwarm', robust=True, cbar=False, square=True, annot=True)

    g.set_xticklabels(labels, rotation=rotation)

    g.set_title("Semantic Similarity", fontsize=25)



def run_and_plot(messages_):

    message_embeddings_ = embed(messages_)

    plot_similarity(messages_, message_embeddings_, 90)
# Some Random Sentences



messages = [

    # Shipping

    "What has happened to my delivery?",

    "What is wrong with my shipping?",



    # Weather

    "What is the weather like tomorrow?",

    "Will it snow tomorrow?",



    # Health

    "An apple a day, keeps the doctors away",

    "Eating apples is healthy",



    # Age

    "How old are you?",

    "what is your age?",

]
# Plot Similarities

run_and_plot(messages)
# Garbage Collection

gc.collect()