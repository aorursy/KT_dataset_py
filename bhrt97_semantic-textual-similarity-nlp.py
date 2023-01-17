import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
input_file = open("/kaggle/input/semantic-textual-similarity-dataset/Microsoft_750_Captions.txt")
output_file = open("/kaggle/input/semantic-textual-similarity-dataset/Microsoft_750_Captions_score.txt")
sent_df=pd.read_csv(input_file, sep='\t', engine='python')
sim_df = pd.read_csv(output_file, sep='\t', engine='python')
micro_df = pd.merge(sent_df,sim_df,left_index=True,right_index=True)
micro_df[:5]
micro_df.rename(columns={'A man is riding a bicycle.':'sent_1',
                        'A man is riding a bike.':'sent_2',
                        '5.000':'sim'},
               inplace=True)

micro_df
import scipy
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# def load_sts_dataset(filename):
#     # Loads a subset of the STS dataset into a DataFrame. In particular both
#     # sentences and their human rated similarity score.
#     sent_pairs = []
#     with tf.io.gfile.GFile(filename, "r") as f:
#         for line in f:
#             ts = line.strip().split("\t")
#             sent_pairs.append((ts[5], ts[6], float(ts[4])))
#     return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


# def download_and_load_sts_data():
#     sts_dataset = tf.keras.utils.get_file(
#         fname="Stsbenchmark.tar.gz",
#         origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
#         extract=True)

#     sts_dev = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
#     sts_test = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

#     return sts_dev, sts_test

# sts_dev, sts_test = download_and_load_sts_data()
# sts_test
import requests

# def download_sick(f): 

#     response = requests.get(f).text

#     lines = response.split("\n")[1:]
#     lines = [l.split("\t") for l in lines if len(l) > 0]
#     lines = [l for l in lines if len(l) == 5]

#     df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
#     df['sim'] = pd.to_numeric(df['sim'])
#     return df
    
# sick_train = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
# sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
# sick_test = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
# sick_all = sick_train.append(sick_test).append(sick_dev)
# sick_all[:5]
# sick_all['label'].value_counts()
import nltk

STOP = set(nltk.corpus.stopwords.words("english"))

class Sentence:
    
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]
import gensim

from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

PATH_TO_WORD2VEC = os.path.expanduser("/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin")
PATH_TO_GLOVE = os.path.expanduser("/kaggle/input/glove840b300dtxt/glove.840B.300d.txt")

word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)
# tmp_file = "/kaggle/input/glove840b300dtxt/glove.840B.300d.txt"
# glove2word2vec(PATH_TO_GLOVE, tmp_file)
# glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

# glove = Word2Vec(sentences, size=300, window=5, min_count=3, workers=4)
import csv


PATH_TO_FREQUENCIES_FILE = "/kaggle/input/freqqq/doc_frequencies.tsv"
PATH_TO_DOC_FREQUENCIES_FILE = "/kaggle/input/freqqq/frequencies.tsv"

def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader: 
            frequencies[row[0]] = int(row[1])
        
    return frequencies
        
frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE)
doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
doc_frequencies["NUM_DOCS"] = 1288431
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None): 

    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]
    
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
    
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue
        
        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)
        
        weights1 = [tokfreqs1[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
                    for token in tokfreqs1] if doc_freqs else None
        weights2 = [tokfreqs2[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
                    for token in tokfreqs2] if doc_freqs else None
                
        embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims
def run_wmd_benchmark(sentences1, sentences2, model, use_stoplist=False):
    
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
    
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
        
        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = [token for token in sent1.tokens if token in model]
            tokens2 = [token for token in sent2.tokens if token in model]
            
        sims.append(-model.wmdistance(tokens1, tokens2))
        
    return sims
from sklearn.decomposition import TruncatedSVD

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def run_sif_benchmark(sentences1, sentences2, model, freqs={}, use_stoplist=False, a=0.001): 
    total_freq = sum(freqs.values())
    
    embeddings = []
    
    # SIF requires us to first collect all sentence embeddings and then perform 
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2): 
        
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
        
        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]
        
        weights1 = [a/(a+freqs.get(token,0)/total_freq) for token in tokens1]
        weights2 = [a/(a+freqs.get(token,0)/total_freq) for token in tokens2]
        
        embedding1 = np.average([model[token] for token in tokens1], axis=0, weights=weights1)
        embedding2 = np.average([model[token] for token in tokens2], axis=0, weights=weights2)
        
        embeddings.append(embedding1)
        embeddings.append(embedding2)
        
    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx*2].reshape(1, -1), 
                              embeddings[idx*2+1].reshape(1, -1))[0][0] 
            for idx in range(int(len(embeddings)/2))]

    return sims
!wget -nc https://raw.githubusercontent.com/facebookresearch/SentEval/master/examples/infersent.py
!wget -nc https://dl.fbaipublicfiles.com/infersent/infersent1.pkl

# import torch

# infersent = torch.load('infersent1.pkl', map_location=lambda storage, loc: storage)
# infersent.use_cuda = False
# torch.nn.Module.dump_patches = True
# infersent.set_w2v_path(PATH_TO_GLOVE)
def run_inf_benchmark(sentences1, sentences2):
    
    raw_sentences1 = [sent1.raw for sent1 in sentences1]
    raw_sentences2 = [sent2.raw for sent2 in sentences2]
    
    infersent.build_vocab(raw_sentences1 + raw_sentences2, tokenize=True)
    embeddings1 = infersent.encode(raw_sentences1, tokenize=True)
    embeddings2 = infersent.encode(raw_sentences2, tokenize=True)
    
    inf_sims = []
    for (emb1, emb2) in zip(embeddings1, embeddings2): 
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        inf_sims.append(sim)

    return inf_sims
import tensorflow as tf 
print(tf.__version__)
import logging
import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
# !pip install "tensorflow_hub>=0.6.0"
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/4")
# import tensorflow_hub as hub

# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
import tensorflow as tf

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def run_gse_benchmark(sentences1, sentences2):
    sts_input1 = tf.placeholder(tf.string, shape=(None))
    sts_input2 = tf.placeholder(tf.string, shape=(None))

    sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
    sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))
        
    sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
      
        [gse_sims] = session.run(
            [sim_scores],
            feed_dict={
                sts_input1: [sent1.raw for sent1 in sentences1],
                sts_input2: [sent2.raw for sent2 in sentences2]
            })
    return gse_sims
def run_experiment(df, benchmarks): 
    
    sentences1 = [Sentence(s) for s in df['sent_1']]
    sentences2 = [Sentence(s) for s in df['sent_2']]
    
    pearson_cors, spearman_cors = [], []
    for label, method in benchmarks:
        sims = method(sentences1, sentences2)
        pearson_correlation = scipy.stats.pearsonr(sims, df['sim'])[0]
        print(label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        spearman_correlation = scipy.stats.spearmanr(sims, df['sim'])[0]
        spearman_cors.append(spearman_correlation)
        
    return pearson_cors, spearman_cors
import functools as ft

benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False)),
              ("AVG-W2V-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True)),
              ("AVG-W2V-TFIDF", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False, doc_freqs=doc_frequencies)),
              ("AVG-W2V-TFIDF-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True, doc_freqs=doc_frequencies)),
#               ("AVG-GLOVE", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False)),
#               ("AVG-GLOVE-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True)),
#               ("AVG-GLOVE-TFIDF", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False, doc_freqs=doc_frequencies)),
#               ("AVG-GLOVE-TFIDF-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True, doc_freqs=doc_frequencies)),
              ("WMD-W2V", ft.partial(run_wmd_benchmark, model=word2vec, use_stoplist=False)), 
              ("WMD-W2V-STOP", ft.partial(run_wmd_benchmark, model=word2vec, use_stoplist=True)), 
#               ("WMD-GLOVE", ft.partial(run_wmd_benchmark, model=glove, use_stoplist=False)), 
#               ("WMD-GLOVE-STOP", ft.partial(run_wmd_benchmark, model=glove, use_stoplist=True)), 
              ("SIF-W2V", ft.partial(run_sif_benchmark, freqs=frequencies, model=word2vec, use_stoplist=False)),
#               ("SIF-GLOVE", ft.partial(run_sif_benchmark, freqs=frequencies, model=glove, use_stoplist=False)), 
#               ("INF", run_inf_benchmark),
              ("GSE", run_gse_benchmark)
             ]
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
pearson_results, spearman_results = {}, {}
pearson_results["SICK-DEV"], spearman_results["SICK-DEV"] = run_experiment(micro_df, benchmarks)
print(" ")
pearson_results["SICK-TEST"], spearman_results["SICK-TEST"] = run_experiment(micro_df, benchmarks)
print(" ")
pearson_results["STS-DEV"], spearman_results["STS-DEV"] = run_experiment(micro_df, benchmarks)
print(" ")
pearson_results["STS-TEST"], spearman_results["STS-TEST"] = run_experiment(micro_df, benchmarks)