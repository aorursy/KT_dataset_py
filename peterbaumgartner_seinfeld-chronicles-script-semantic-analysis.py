from functools import partial
from itertools import combinations
from textwrap import wrap

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.downloader as api
import plotly.graph_objs as go
import spacy
import umap
from gensim.corpora import Dictionary
from gensim.matutils import softcossim
from plotly.offline import init_notebook_mode, iplot
%%capture

nlp = spacy.load("en")

model = api.load("glove-wiki-gigaword-50");
model.init_sims(replace=True)
scripts = pd.read_csv("../input/scripts.csv", index_col=0)
scripts.head()
scripts.tail()
character = "KRAMER"
character_script = scripts[(scripts["Character"] == character)]
print(f"original n lines: {len(scripts)}, character n lines {len(character_script)}")
%%time

dialogues = character_script["Dialogue"].astype(str).tolist()

tokenized_docs = []
for i, doc in enumerate(nlp.pipe(dialogues, n_threads=-1)):
    tokens = [token.lower_ for token in doc if token.lower_ in model.vocab and token.is_alpha]
    if len(tokens) > 0:
        tokenized_docs.append((i, tokens))

d_index, d_tokenized = zip(*tokenized_docs)
kept_dialogues = [dialogues[i] for i in d_index]
print(f"original: {len(dialogues)}, reduced: {len(kept_dialogues)}")
tfidf_vectors = TfidfVectorizer(analyzer=lambda x: x).fit_transform(d_tokenized)
cos_dist = 1 - (tfidf_vectors.toarray() * tfidf_vectors.T)
tfidf_embedding = umap.UMAP(metric="precomputed", random_state=666).fit_transform(cos_dist)
embedding_df = pd.DataFrame(tfidf_embedding, columns=["dim0", "dim1"])
sentence_text_series = pd.Series(kept_dialogues, name="text")
sentence_token_series = pd.Series(d_tokenized, name="tokens")
tfidf_df = pd.concat([sentence_text_series, sentence_token_series, embedding_df], axis=1)
def build_tooltip(row):
    text = "<br>".join(wrap(row["text"], 40))
    tokens = "<br>".join(wrap(", ".join(row["tokens"]), 40))
    full_string = [
        "<b>Text:</b> ",
        text,
        "<br>",
        "<b>Tokens:</b> ",
        tokens
    ]
    return "".join(full_string)

tfidf_df["tooltip"] = tfidf_df.apply(build_tooltip, axis=1)
init_notebook_mode(connected=True)

trace = go.Scatter(
    x = tfidf_df["dim0"],
    y = tfidf_df["dim1"],
    name = "TFIDF Embedding",
    mode = "markers",
    marker = dict(
        color = "rgba(49, 76, 182, .8)",
        size = 5,
        line = dict(width=1)),
    text=tfidf_df["tooltip"])

layout = dict(title="2D Embeddings - TFIDF",
             yaxis = dict(zeroline=False),
             xaxis = dict(zeroline=False),
             hovermode = "closest")

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
%%time

dictionary = Dictionary(d_tokenized)
corpus = [dictionary.doc2bow(document) for document in d_tokenized]
similarity_matrix = model.similarity_matrix(dictionary)
corpus_softcossim = partial(softcossim, similarity_matrix=similarity_matrix)
%%time

sentence_pairs = combinations(corpus, 2)
scs_sims = [corpus_softcossim(d1, d2) for d1, d2 in sentence_pairs]
n_sentences = len(corpus)
scs_empty = np.zeros((n_sentences, n_sentences))
upper_indices = np.triu_indices(n_sentences, 1)
scs_empty[upper_indices] = scs_sims
scs_sim = np.triu(scs_empty, -1).T + scs_empty
np.fill_diagonal(scs_sim, 1)
scs_dist = 1 - scs_sim
scs_embedding = umap.UMAP(metric="precomputed", random_state=666).fit_transform(scs_dist)
scs_embedding_df = pd.DataFrame(scs_embedding, columns=["dim0", "dim1"])
scs_df = pd.concat([sentence_text_series, sentence_token_series, scs_embedding_df], axis=1)
scs_df["tooltip"] = scs_df.apply(build_tooltip, axis=1)
trace = go.Scatter(
    x = scs_df["dim0"],
    y = scs_df["dim1"],
    name = "SCS Embedding",
    mode = "markers",
    marker = dict(
        color = "rgba(49, 76, 182, .8)",
        size = 5,
        line = dict(width=1)),
    text=scs_df["tooltip"])

layout = dict(title="2D Embeddings - SCS",
             yaxis = dict(zeroline=False),
             xaxis = dict(zeroline=False),
             hovermode = "closest")

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
