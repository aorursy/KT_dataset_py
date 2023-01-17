import os

import re

import html as ihtml

import warnings

import random

warnings.filterwarnings('ignore')



os.environ["TFHUB_CACHE_DIR"] = "/tmp/"



import spacy

nlp = spacy.load('en_core_web_sm')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')

#nlp.remove_pipe('tagger')



import pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()



from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import gensim

import scipy



import tensorflow as tf

import tensorflow_hub as hub



import matplotlib.pyplot as plt

import seaborn as sns

import umap



pd.set_option('display.max_colwidth', -1)



SEED = 13

random.seed(SEED)

np.random.seed(SEED)



%matplotlib inline
input_dir = '../input/'



questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))
# Spacy Tokenfilter for part-of-speech tagging

token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']



def clean_text(text, remove_hashtags=True):

    text = BeautifulSoup(ihtml.unescape(text), "lxml").text

    text = re.sub(r"http[s]?://\S+", "", text)

    if remove_hashtags:

        text = re.sub(r"#[a-zA-Z\-]+", "", text)

    text = re.sub(r"\s+", " ", text)        

    return text



def nlp_preprocessing(data):

    """ Use NLP to transform the text corpus to cleaned sentences and word tokens



    """    

    def token_filter(token):

        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list



        """    

        return not token.is_stop and token.is_alpha and token.pos_ in token_pos

    

    processed_tokens = []

    data_pipe = nlp.pipe(data, n_threads=4)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        processed_tokens.append(filtered_tokens)

    return processed_tokens
questions['questions_full_text'] = questions['questions_title'] + ' '+ questions['questions_body']
sample_text = questions[questions['questions_full_text'].str.contains("&a")]["questions_full_text"].iloc[0]

sample_text
sample_text = clean_text(sample_text)

sample_text
sample = nlp_preprocessing([sample_text])

" ".join(sample[0])
%%time

questions['questions_full_text'] = questions['questions_full_text'].apply(clean_text)
questions['questions_full_text'].sample(2)
%%time

questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])
questions['nlp_tokens'].sample(2)
# Gensim Dictionary

extremes_no_below = 10

extremes_no_above = 0.6

extremes_keep_n = 8000



# LDA

num_topics = 10

passes = 20

chunksize = 1000

alpha = 1/50
def get_model_results(ldamodel, corpus, dictionary):

    """ Create doc-topic probabilities table and visualization for the LDA model



    """  

    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)

    transformed = ldamodel.get_document_topics(corpus)

    df = pd.DataFrame.from_records([{v:k for v, k in row} for row in transformed])

    return vis, df  
%%time

lda_tokens = questions['nlp_tokens']



# Gensim Dictionary

lda_dic = gensim.corpora.Dictionary(lda_tokens)

lda_dic.filter_extremes(no_below=extremes_no_below, no_above=extremes_no_above, keep_n=extremes_keep_n)

lda_corpus = [lda_dic.doc2bow(doc) for doc in lda_tokens]



lda_tfidf = gensim.models.TfidfModel(lda_corpus)

lda_corpus = lda_tfidf[lda_corpus]



# Create LDA Model

lda_model = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=num_topics, 

                                            id2word = lda_dic, passes=passes,

                                            chunksize=chunksize,update_every=0,

                                            alpha=alpha, random_state=SEED)
# Create Visualization and Doc-Topic Probapilities

lda_vis, lda_result = get_model_results(lda_model, lda_corpus, lda_dic)

lda_questions = questions[['questions_id', 'questions_title', 'questions_body']]

lda_questions = pd.concat([lda_questions, lda_result.add_prefix('Topic_')], axis=1)
# Disabled for compatibility issue

# lda_vis
print("\n\n".join(["Topic{}:\n {}".format(i, j) for i, j in lda_model.print_topics()]))
corpus_csr = gensim.matutils.corpus2csc(lda_corpus).T
# There exist some zero rows:

non_zeros = np.where(corpus_csr.sum(1) != 0)[0]

print(corpus_csr.shape[0])

corpus_csr = corpus_csr[non_zeros, :]

print(corpus_csr.shape[0])
# Normalize by row

corpus_csr = corpus_csr.multiply(

    scipy.sparse.csr_matrix(1/np.sqrt(corpus_csr.multiply(corpus_csr).sum(1))))
# Double check the norms

np.sum(np.abs(corpus_csr.multiply(corpus_csr).sum(1) - 1) > 0.001)
%%time

embedding = umap.UMAP(metric="cosine", n_components=2).fit_transform(corpus_csr)
df_emb = pd.DataFrame(embedding, columns=["x", "y"])

df_emb["label"] = np.argmax(lda_result.iloc[non_zeros].fillna(0).values, axis=1)
df_emb_sample = df_emb.sample(5000)

fig, ax = plt.subplots(figsize=(12, 10))

plt.scatter(

    df_emb_sample["x"].values, df_emb_sample["y"].values, s=2, c=df_emb_sample["label"].values# , cmap="Spectral"

)

plt.setp(ax, xticks=[], yticks=[])

cbar = plt.colorbar(boundaries=np.arange(11)-0.5)

cbar.set_ticks(np.arange(10))

plt.title("TF-IDF matrix embedded into two dimensions by UMAP", fontsize=18)

plt.show()
g = sns.FacetGrid(df_emb, col="label", col_wrap=2, height=5, aspect=1)

g.map(plt.scatter, "x", "y", s=0.2).fig.subplots_adjust(wspace=.05, hspace=.5)
# keep well separated points

df_emb_sample = df_emb[np.amax(lda_result.iloc[non_zeros].fillna(0).values, axis=1) > 0.7]

print("Before:", df_emb.shape[0], "After:", df_emb_sample.shape[0])

g = sns.FacetGrid(df_emb_sample, col="label", col_wrap=2, height=5, aspect=1)

g.map(plt.scatter, "x", "y", s=0.3).fig.subplots_adjust(wspace=.05, hspace=.5)
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
import logging

from tqdm import tqdm_notebook

tf.logging.set_verbosity(logging.WARNING)

BATCH_SIZE = 128



sentence_input = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than

# approximately normalized.

sentence_emb = tf.nn.l2_normalize(embed(sentence_input), axis=1)



sentence_embeddings = []       

with tf.Session() as session:

    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    for i in tqdm_notebook(range(0, len(questions), BATCH_SIZE)):

        sentence_embeddings.append(

            session.run(

                sentence_emb, 

                feed_dict={

                    sentence_input: questions["questions_full_text"].iloc[i:(i+BATCH_SIZE)].values

                }

            )

        )
sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)

sentence_embeddings.shape
%%time

import umap

embedding = umap.UMAP(metric="cosine", n_components=2).fit_transform(sentence_embeddings)
df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])

df_se_emb["label"] = np.argmax(lda_result.fillna(0).values, axis=1)

df_se_emb["label"] = df_se_emb["label"].astype("category")
df_emb_sample = df_se_emb.sample(5000)

fig, ax = plt.subplots(figsize=(12, 10))

plt.scatter(

    df_emb_sample["x"].values, df_emb_sample["y"].values, s=2, c=df_emb_sample["label"].values# , cmap="Spectral"

)

plt.setp(ax, xticks=[], yticks=[])

cbar = plt.colorbar(boundaries=np.arange(11)-0.5)

cbar.set_ticks(np.arange(10))

plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)

plt.show()
g = sns.FacetGrid(df_se_emb, col="label", col_wrap=2, height=5, aspect=1)

g.map(plt.scatter, "x", "y", s=0.2).fig.subplots_adjust(wspace=.05, hspace=.5)
def find_similar(idx, top_k):

    cosine_similarities = sentence_embeddings @ sentence_embeddings[idx][:, np.newaxis]

    return np.argsort(cosine_similarities[:, 0])[::-1][1:(top_k+1)]
IDX = 0

similar_ids = find_similar(IDX, top_k=3).tolist()

for idx in [IDX] + similar_ids:

    print(questions["questions_full_text"].iloc[idx], "\n")
IDX = 5

similar_ids = find_similar(IDX, top_k=3).tolist()

for idx in [IDX] + similar_ids:

    print(questions["questions_full_text"].iloc[idx], "\n")
IDX = 522

similar_ids = find_similar(IDX, top_k=3).tolist()

for idx in [IDX] + similar_ids:

    print(questions["questions_full_text"].iloc[idx], "\n")
IDX = 13331

similar_ids = find_similar(IDX, top_k=3).tolist()

for idx in [IDX] + similar_ids:

    print(questions["questions_full_text"].iloc[idx], "\n")