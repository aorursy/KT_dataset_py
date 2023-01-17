from collections import Counter
import numpy as np
import nltk
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v
data = pd.read_csv('../input/winemag-data_first150k.csv')
labels = data['variety']
descriptions = data['description']
print('{}   :   {}'.format(labels.tolist()[0], descriptions.tolist()[0]))
print('{}   :   {}'.format(labels.tolist()[56], descriptions.tolist()[56]))
print('{}   :   {}'.format(labels.tolist()[93], descriptions.tolist()[93]))
varietal_counts = labels.value_counts()
print(varietal_counts[:50])
corpus_raw = ""
for description in descriptions:
    corpus_raw += description
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))
print(raw_sentences[234])
print(sentence_to_wordlist(raw_sentences[234]))
token_count = sum([len(sentence) for sentence in sentences])
print('The wine corpus contains {0:,} tokens'.format(token_count))
num_features = 300
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993
wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
wine2vec.build_vocab(sentences)
print('Word2Vec vocabulary length:', len(wine2vec.wv.vocab))
print(wine2vec.corpus_count)
wine2vec.train(sentences, total_examples=wine2vec.corpus_count, epochs=wine2vec.iter)
wine2vec.most_similar('melon')
wine2vec.most_similar('berry')
wine2vec.most_similar('oak')
wine2vec.most_similar('acidic')
wine2vec.most_similar('full')
wine2vec.most_similar('tannins')
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = wine2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2
nearest_similarity_cosmul('oak', 'vanilla', 'cherry');
nearest_similarity_cosmul('full', 'berry', 'light');
nearest_similarity_cosmul('tannins', 'plum', 'fresh');
nearest_similarity_cosmul('full', 'bodied', 'acidic');