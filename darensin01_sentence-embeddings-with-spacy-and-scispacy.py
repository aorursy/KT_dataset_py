!pip install scispacy --quiet

!pip install spacy --quiet

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz --quiet

!pip install wordcloud --quiet
import glob

import json

import pandas as pd

import numpy as np

import scispacy

import spacy

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from itertools import chain

from collections import Counter

from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

from pprint import pprint

from tqdm.notebook import trange, tqdm

tqdm.pandas()

%matplotlib inline
nlp = spacy.load("en_core_sci_md")
result = []

for filename in tqdm(glob.glob("../input/CORD-19-research-challenge/**/*.json", recursive=True)):

    with open(filename, encoding='utf-8') as f:

        json_file = json.load(f)

        

    body_texts = json_file.get("body_text", [])

    body_text = " ".join([b['text'] for b in body_texts])

    

    sections = [b['section'].strip().lower() for b in body_texts if (b['section'].strip() != "" and not b['section'].isdigit())]

    sections = list(set(sections))

    

    if len(sections) == 0:

        sections = ""

    

    sha = filename.split("/")[-1].split(".json")[0]

    

    result.append((sha, sections, body_text))
papers_df = pd.DataFrame(result, columns=["sha", "sections", "body_text"])

len(papers_df)
source_metadata_df = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
df = pd.merge(papers_df, source_metadata_df, on="sha")
df = pd.read_json("../input/cord19-embeddings-with-scispacy/covid_with_embedding.json")

df = df[['sha', 'sections', 'body_text', 'title', 'doi', 'abstract', 'body_text_embedding', 'body_text_gensim_keywords']]
query_word = "misinformation"

query_vector = None



if query_word in nlp.vocab:

    query_vector = nlp.vocab[query_word].vector
def cosine_distance_from_query(e):

    embedding = np.asarray(e)

    return cosine_distances(embedding.reshape(1, -1), query_vector.reshape(1, -1))[0][0]



df['dist_to_query_vector'] = df['body_text_embedding'].progress_apply(cosine_distance_from_query)
N = 10

top_N_similar_rows = df.loc[df['dist_to_query_vector'].sort_values().head(10).index]
top_N_similar_rows[['sha', 'body_text', 'title', 'doi', 'body_text_gensim_keywords']]
query_counter = Counter(chain.from_iterable(top_N_similar_rows['body_text_gensim_keywords'].apply(set).values))

query_wordcloud = WordCloud().generate_from_frequencies(query_counter)

plt.figure(figsize=(15, 8))

plt.imshow(query_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def manhattan_distance_from_query(e):

    embedding = np.asarray(e)

    return manhattan_distances(embedding.reshape(1, -1), query_vector.reshape(1, -1))[0][0]



df['man_dist_to_query_vector'] = df['body_text_embedding'].progress_apply(manhattan_distance_from_query)

top_N_similar_rows_manhattan = df.loc[df['man_dist_to_query_vector'].sort_values().head(10).index]

top_N_similar_rows_manhattan[['sha', 'body_text', 'title', 'doi', 'body_text_gensim_keywords']]
query_counter = Counter(chain.from_iterable(top_N_similar_rows_manhattan['body_text_gensim_keywords'].apply(set).values))

query_wordcloud = WordCloud().generate_from_frequencies(query_counter)

plt.figure(figsize=(15, 8))

plt.imshow(query_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()