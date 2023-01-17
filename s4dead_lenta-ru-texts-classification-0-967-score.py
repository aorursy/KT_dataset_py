# !pip install pymorphy2

# !git clone https://github.com/facebookresearch/fastText.git
# !pip install fastText/.
import os
import re
from collections import defaultdict
import pickle
import random

from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

from sklearn.metrics import precision_score, recall_score, \
    f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
import fasttext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader

from google.colab import drive

import nltk

nltk.download('stopwords')
nltk.download('punkt')

%matplotlib inline
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
set_seed(12345)
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 800)
if_use_gpu = torch.cuda.is_available()
if_use_gpu
FloatTensor = torch.cuda.FloatTensor if if_use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if if_use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if if_use_gpu else torch.ByteTensor
Tensor = FloatTensor
drive.mount('/content/gdrive', force_remount=True)
# !unzip -q /content/gdrive/My\ Drive/data.zip -d ./
def create_dirs(dirs_names):
    for dir_name in dirs_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
data_dir = 'data'

create_dirs([data_dir])
preprocessed_dir = os.path.join(data_dir, 'preprocessed')
ft_dir = os.path.join(data_dir, 'ft')
w2v_dir = os.path.join(data_dir, 'w2v')
cnn_dir = os.path.join(data_dir, 'CNN')
best_dir = os.path.join(data_dir, 'best')

create_dirs([preprocessed_dir, ft_dir, w2v_dir, cnn_dir, best_dir])
n_classes = 5
data = pd.read_csv(os.path.join(data_dir, 'lenta-ru-train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'lenta-ru-test.csv'))
data.head(3)
test_data.head(3)
data.isna().sum()
test_data.isna().sum()
data = data.dropna()
n_samples = data.shape[0]
n_test_samples = test_data.shape[0]
n_samples, n_test_samples
data['topic'].value_counts()
regex = re.compile("[А-Яа-яA-z]+")

def words_only(text, regex=regex):
    try:
        return regex.findall(text)
    except:
        return []
m = MorphAnalyzer()

@lru_cache(maxsize=256)
def lemmatize_token(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form

def lemmatize_tokens(tokens):
    return [lemmatize_token(token) for token in tokens]
stops = stopwords.words('russian')

def remove_stopwords(tokens, stops=stops):
    return [token for token in tokens if not token in stops and len(token) > 2]
def sentence2tokens(sentence, if_remove_stopwords):
    tokens = words_only(sentence)
    lemmas = lemmatize_tokens(tokens)
    if if_remove_stopwords:
        lemmas = remove_stopwords(lemmas)
    return lemmas
tokenize = sent_tokenize

def text2tokens_in_sentences(text, sent_tokenize_func=tokenize, if_remove_stopwords=True):
    striped_text = text.strip()
    sentences = sent_tokenize_func(striped_text)
    tokens_in_sentences = []
    for sentence in sentences:
        if len(sentence) > 0:
            lowered_sentence = sentence.lower()
            tokens = sentence2tokens(lowered_sentence, if_remove_stopwords)
            if tokens:
                tokens_in_sentences.append(tokens)
    return tokens_in_sentences
n_samples_1 = n_samples // 2
data_1 = data[:n_samples_1]
with Pool(8) as p:
    prep_titles_tokens_in_sentences_1 = \
        list(tqdm(p.imap(text2tokens_in_sentences, data_1['title']), total=n_samples_1))
print(*prep_titles_tokens_in_sentences_1[:5])
with Pool(8) as p:
    prep_texts_tokens_in_sentences_1 = \
        list(tqdm(p.imap(text2tokens_in_sentences, data_1['text']), total=n_samples_1))
print(*prep_texts_tokens_in_sentences_1[:3])
data_1['prep_title_tokens_in_sentences'] = prep_titles_tokens_in_sentences_1
data_1['prep_text_tokens_in_sentences'] = prep_texts_tokens_in_sentences_1

mask = data_1['prep_title_tokens_in_sentences'].astype(bool) & \
    data_1['prep_text_tokens_in_sentences'].astype(bool)
data_1 = data_1[mask]

prep_titles_tokens_in_sentences_1 = data_1['prep_title_tokens_in_sentences'].tolist()
prep_texts_tokens_in_sentences_1 = data_1['prep_text_tokens_in_sentences'].tolist()
# with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences_1.pkl'), 'wb') as outfile:
#     pickle.dump(data_1['prep_title_tokens_in_sentences'].tolist(), outfile)
# with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences_1.pkl'), 'wb') as outfile:
#     pickle.dump(data_1['prep_text_tokens_in_sentences'].tolist(), outfile)
# data_1.to_csv(os.path.join(preprocessed_dir, 'data_1.csv'),
#               columns=['title', 'text', 'topic', 'topic_label'],
#               index=False)
n_samples_2 = n_samples - n_samples // 2
data_2 = data[-n_samples_2:]
with Pool(8) as p:
    prep_titles_tokens_in_sentences_2 = \
        list(tqdm(p.imap(text2tokens_in_sentences, data_2['title']), total=n_samples_2))
print(*prep_titles_tokens_in_sentences_2[:5])
with Pool(8) as p:
    prep_texts_tokens_in_sentences_2 = \
        list(tqdm(p.imap(text2tokens_in_sentences, data_2['text']), total=n_samples_2))
print(*prep_texts_tokens_in_sentences_2[:3])
data_2['prep_title_tokens_in_sentences'] = prep_titles_tokens_in_sentences_2
data_2['prep_text_tokens_in_sentences'] = prep_texts_tokens_in_sentences_2

mask = data_2['prep_title_tokens_in_sentences'].astype(bool) & \
    data_2['prep_text_tokens_in_sentences'].astype(bool)
data_2 = data_2[mask]

prep_titles_tokens_in_sentences_2 = data_2['prep_title_tokens_in_sentences'].tolist()
prep_texts_tokens_in_sentences_2 = data_2['prep_text_tokens_in_sentences'].tolist()
# with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences_2.pkl'), 'wb') as outfile:
#     pickle.dump(data_2['prep_title_tokens_in_sentences'].tolist(), outfile)
# with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences_2.pkl'), 'wb') as outfile:
#     pickle.dump(data_2['prep_text_tokens_in_sentences'].tolist(), outfile)
# data_2.to_csv(os.path.join(preprocessed_dir, 'data_2.csv'),
#               columns=['title', 'text', 'topic', 'topic_label'],
#               index=False)
with Pool(8) as p:
    prep_test_titles_tokens_in_sentences = \
        list(tqdm(p.imap(text2tokens_in_sentences, test_data['title']), total=n_test_samples))
print(*prep_test_titles_tokens_in_sentences[:5])
with Pool(8) as p:
    prep_test_texts_tokens_in_sentences = \
        list(tqdm(p.imap(text2tokens_in_sentences, test_data['text']), total=n_test_samples))
print(*prep_test_texts_tokens_in_sentences[:3])
test_data['prep_title_tokens_in_sentences'] = prep_test_titles_tokens_in_sentences
test_data['prep_text_tokens_in_sentences'] = prep_test_texts_tokens_in_sentences
# with open(os.path.join(preprocessed_dir, 'test_titles_tokens_in_sentences.pkl'), 'wb') as outfile:
#     pickle.dump(test_data['prep_title_tokens_in_sentences'].tolist(), outfile)
# with open(os.path.join(preprocessed_dir, 'test_texts_tokens_in_sentences.pkl'), 'wb') as outfile:
#     pickle.dump(test_data['prep_text_tokens_in_sentences'].tolist(), outfile)
# test_data.to_csv(os.path.join(preprocessed_dir, 'test_data.csv'),
#                  columns=['title', 'text'],
#                  index=False)
with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences_1.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences_1 = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences_1.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences_1 = pickle.load(infile)
data_1 = pd.read_csv(os.path.join(preprocessed_dir, 'data_1.csv'))

with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences_2.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences_2 = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences_2.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences_2 = pickle.load(infile)
data_2 = pd.read_csv(os.path.join(preprocessed_dir, 'data_2.csv'))
prep_titles_tokens_in_sentences = prep_titles_tokens_in_sentences_1 + prep_titles_tokens_in_sentences_2
prep_texts_tokens_in_sentences = prep_texts_tokens_in_sentences_1 + prep_texts_tokens_in_sentences_2

data = data_1.append(data_2)
data = data.reset_index(drop=True)
# with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences.pkl'), 'wb') as outfile:
#     pickle.dump(prep_titles_tokens_in_sentences, outfile)
# with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences.pkl'), 'wb') as outfile:
#     pickle.dump(prep_texts_tokens_in_sentences, outfile)
# data.to_csv(os.path.join(preprocessed_dir, 'data.csv'),
#             columns=['title', 'text', 'topic', 'topic_label'],
#             index=False)
with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences = pickle.load(infile)

with open(os.path.join(preprocessed_dir, 'test_titles_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_test_titles_tokens_in_sentences = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'test_texts_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_test_texts_tokens_in_sentences = pickle.load(infile)
n_samples = len(prep_titles_tokens_in_sentences)
n_test_samples = len(prep_test_titles_tokens_in_sentences)
prep_titles = []
prep_texts = []

for i in range(n_samples):
    prep_title_tokens_in_sentences = prep_titles_tokens_in_sentences[i]

    prep_title_sentences = []

    for prep_title_tokens_in_sentence in prep_title_tokens_in_sentences:
        prep_title_sentences.append(' '.join(prep_title_tokens_in_sentence))

    prep_title = ' '.join(prep_title_sentences)

    prep_titles.append(prep_title)

    prep_text_tokens_in_sentences = prep_texts_tokens_in_sentences[i]

    prep_text_sentences = []

    for prep_text_tokens_in_sentence in prep_text_tokens_in_sentences:
        prep_text_sentences.append(' '.join(prep_text_tokens_in_sentence))
    
    prep_text = ' '.join(prep_text_sentences)

    prep_texts.append(prep_text)
prep_test_titles = []
prep_test_texts = []

for i in range(n_test_samples):
    prep_test_title_tokens_in_sentences = prep_test_titles_tokens_in_sentences[i]

    prep_test_title_sentences = []

    for prep_test_title_tokens_in_sentence in prep_test_title_tokens_in_sentences:
        prep_test_title_sentences.append(' '.join(prep_test_title_tokens_in_sentence))

    prep_test_title = ' '.join(prep_test_title_sentences)

    prep_test_titles.append(prep_test_title)

    prep_test_text_tokens_in_sentences = prep_test_texts_tokens_in_sentences[i]

    prep_test_text_sentences = []

    for prep_test_text_tokens_in_sentence in prep_test_text_tokens_in_sentences:
        prep_test_text_sentences.append(' '.join(prep_test_text_tokens_in_sentence))
    
    prep_test_text = ' '.join(prep_test_text_sentences)

    prep_test_texts.append(prep_test_text)
# with open(os.path.join(preprocessed_dir, 'titles.pkl'), 'wb') as outfile:
#     pickle.dump(prep_titles, outfile)
# with open(os.path.join(preprocessed_dir, 'texts.pkl'), 'wb') as outfile:
#     pickle.dump(prep_texts, outfile)

# with open(os.path.join(preprocessed_dir, 'test_titles.pkl'), 'wb') as outfile:
#     pickle.dump(prep_test_titles, outfile)
# with open(os.path.join(preprocessed_dir, 'test_texts.pkl'), 'wb') as outfile:
#     pickle.dump(prep_test_texts, outfile)
y = pd.read_csv(os.path.join(preprocessed_dir, 'data.csv'), usecols=['topic_label'])['topic_label'].values
n_samples = y.size
n_samples
inds = np.arange(n_samples)
inds_train, inds_val = train_test_split(inds, test_size=0.15, random_state=12345)
y_train = y[inds_train]
y_val = y[inds_val]
n_train_samples = y_train.size
n_val_samples = y_val.size
n_train_samples, n_val_samples
# на каких данных обучаться [0 - тексты, 2 - заголовки, 1 - и то и другое]
flag_data_train = 1
with open(os.path.join(preprocessed_dir, 'texts.pkl'), 'rb') as infile:
    texts = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'titles.pkl'), 'rb') as infile:
    titles = pickle.load(infile)
if flag_data_train == 0:
    X = texts
elif flag_data_train == 2:
    X = titles
else:
    X = list(map(' '.join, zip(titles, texts)))
len(X)
X_train = [X[ind] for ind in inds_train]
X_val = [X[ind] for ind in inds_val]
with open(os.path.join(ft_dir, 'data.train.txt'), 'w+') as train_outfile:
    for i in range(n_train_samples):
        train_outfile.write(f'__label__{y_train[i]} {X_train[i]}\n')

with open(os.path.join(ft_dir, 'val.txt'), 'w+') as val_outfile:
    for i in range(n_val_samples):
        val_outfile.write(f'__label__{y_val[i]} {X_val[i]}\n')
%time ft_classifier = fasttext.train_supervised(os.path.join(ft_dir, 'data.train.txt'))  # , 'model')
# где будет лежать обученный классификатор
ft_classifier_file_name = 'classifier.bin'
# ft_classifier.save_model(os.path.join(ft_dir, ft_classifier_file_name))
ft_classifier = fasttext.load_model(os.path.join(ft_dir, ft_classifier_file_name))
y_val_pred = ft_classifier.predict(X_val)[0]
y_val_pred = [label[0].split('_')[-1] for label in y_val_pred]
y_val_pred = np.array(y_val_pred, dtype=int)
def print_results(N, p, r):
    print('N\t' + str(N))
    print('P@{}\t{:.3f}'.format(1, p))  # precision
    print('R@{}\t{:.3f}'.format(1, r))  # recall
result = ft_classifier.test(os.path.join(ft_dir, 'val.txt'))
print_results(*result)
labels = range(5)
print('Precision: {0:6.2f}'.format(precision_score(y_val, y_val_pred, average='macro')))
print('Recall: {0:6.2f}'.format(recall_score(y_val, y_val_pred, average='macro')))
print('F1-measure: {0:6.2f}'.format(f1_score(y_val, y_val_pred, average='macro')))
print('Accuracy: {0:6.2f}'.format(accuracy_score(y_val, y_val_pred)))
print(classification_report(y_val, y_val_pred))  # [i[0] for i in pred]))

sns.heatmap(data=confusion_matrix(y_val, y_val_pred),
            annot=True, fmt="d", cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion matrix')
plt.show()
# на каких данных обучаться [0 - тексты, 2 - заголовки, 1 - и то и другое]
flag_data_train = 1
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences = pickle.load(infile)
X = []
if flag_data_train >= 1:
    for tokens_in_sentences in prep_titles_tokens_in_sentences:
        for tokens_in_sentence in tokens_in_sentences:
            X.append(tokens_in_sentence)
if flag_data_train <= 1:
    for tokens_in_sentences in prep_texts_tokens_in_sentences:
        for tokens_in_sentence in tokens_in_sentences:
            X.append(tokens_in_sentence)
len(X)
# параметры модели
emb_dim = 300
window = 5
min_count = 5
workers = 4
sample = 1e-3
%%time
w2v_model = Word2Vec(X, size=emb_dim, window=window, min_count=min_count,
                     workers=workers, sample=sample)
# куда сохранять обученную модель
save_w2v_model_file_name = 'model.bin'
# w2v_model.save(os.path.join(w2v_dir, save_w2v_model_file_name))
# откуда брать обученную модель
load_w2v_model_file_name = 'model.bin'
w2v_model = Word2Vec.load(os.path.join(w2v_dir, load_w2v_model_file_name))
w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.vectors))
# на каких данных обучаться [0 - тексты, 2 - заголовки, 1 - и то и другое]
flag_data_train = 1
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences = pickle.load(infile)
X = []
for title_tokens_in_sentences, text_tokens_in_sentences in zip(prep_titles_tokens_in_sentences, prep_texts_tokens_in_sentences):
    tokens = []
    if flag_data_train >= 1:
        for tokens_in_sentence in title_tokens_in_sentences:
            tokens += tokens_in_sentence
    if flag_data_train <= 1:
        for tokens_in_sentence in text_tokens_in_sentences:
            tokens += tokens_in_sentence
    X.append(tokens)
len(X)
X_train = [X[ind] for ind in inds_train]
X_val = [X[ind] for ind in inds_val]
# параметры классификатора
weights = 'distance'
metric = 'cosine'
n_jobs = -1
# Получаем вектор текста из векторов его слов
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf,
                                       [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
knn_w2v_mean = Pipeline([
    ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v)),
    ('k-NN', KNeighborsClassifier(weights=weights, metric=metric, n_jobs=n_jobs))
])

knn_w2v_tfidf = Pipeline([
    ('word2vec vectorizer', TfidfEmbeddingVectorizer(w2v)),
    ('k-NN', KNeighborsClassifier(weights=weights, metric=metric, n_jobs=n_jobs))
])
%time knn_w2v_mean.fit(X_train, y_train)
%time knn_w2v_tfidf.fit(X_train, y_train)
y_val_pred_mean = knn_w2v_mean.predict(X_val)
y_val_pred_tfidf = knn_w2v_tfidf.predict(X_val)
labels = range(5)
print('Precision: {0:6.2f}'.format(precision_score(y_val, y_val_pred_mean, average='macro')))
print('Recall: {0:6.2f}'.format(recall_score(y_val, y_val_pred_mean, average='macro')))
print('F1-measure: {0:6.2f}'.format(f1_score(y_val, y_val_pred_mean, average='macro')))
print('Accuracy: {0:6.2f}'.format(accuracy_score(y_val, y_val_pred_mean)))
print(classification_report(y_val, y_val_pred_mean))  # [i[0] for i in pred]))

sns.heatmap(data=confusion_matrix(y_val, y_val_pred_mean),
            annot=True, fmt="d", cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion matrix')
plt.show()
print('Precision: {0:6.2f}'.format(precision_score(y_val, y_val_pred_tfidf, average='macro')))
print('Recall: {0:6.2f}'.format(recall_score(y_val, y_val_pred_tfidf, average='macro')))
print('F1-measure: {0:6.2f}'.format(f1_score(y_val, y_val_pred_tfidf, average='macro')))
print('Accuracy: {0:6.2f}'.format(accuracy_score(y_val, y_val_pred_tfidf)))
print(classification_report(y_val, y_val_pred_tfidf))  # [i[0] for i in pred]))

sns.heatmap(data=confusion_matrix(y_val, y_val_pred_tfidf),
            annot=True, fmt="d", cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion matrix')
plt.show()
unk_word = '<unk>'
pad_word = '<pad>'
def get_pretrained_embeddings(word2vec, emb_dim):
    w2ind = {}
    embeddings = np.empty((len(word2vec) + 2, emb_dim))

    w2ind[unk_word] = 0
    w2ind[pad_word] = 1
    
    embeddings[w2ind[unk_word]] = np.random.uniform(-0.25, 0.25, emb_dim)
    embeddings[w2ind[pad_word]] = np.zeros(emb_dim)
    
    for ind, (word, vec) in enumerate(word2vec.items()):
        w2ind[word] = ind + 2
        embeddings[w2ind[word]] = vec

    return w2ind, embeddings
emb_dim = len(list(w2v.values())[0])
w2ind, pretrained_embeddings = get_pretrained_embeddings(w2v, emb_dim)
pretrained_embeddings = torch.tensor(pretrained_embeddings)
# на каких данных обучаться [0 - тексты, 2 - заголовки, 1 - и то и другое]
flag_data_train = 1
# Получаем входные данные - обрезаем/заполняем тексты и кодируем слова
def pad_and_encode(texts_tokens, word2ind, text_tokens_len):
    encoded_padded_texts = []
    for text_tokens in texts_tokens:
        text_tokens = text_tokens[:text_tokens_len]
        text_tokens += [pad_word] * (text_tokens_len - len(text_tokens))

        encoded_padded_text = []
        for text_token in text_tokens:
            encoded_token = word2ind[text_token] if text_token in word2ind else word2ind[unk_word]
            encoded_padded_text.append(encoded_token)

        encoded_padded_texts.append(encoded_padded_text)

    return np.array(encoded_padded_texts)
with open(os.path.join(preprocessed_dir, 'texts_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_texts_tokens_in_sentences = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'titles_tokens_in_sentences.pkl'), 'rb') as infile:
    prep_titles_tokens_in_sentences = pickle.load(infile)
X = []
max_text_tokens_len = -1
for title_tokens_in_sentences, text_tokens_in_sentences in zip(prep_titles_tokens_in_sentences, prep_texts_tokens_in_sentences):
    tokens = []
    if flag_data_train >= 1:
        for tokens_in_sentence in title_tokens_in_sentences:
            tokens += tokens_in_sentence
    if flag_data_train <= 1:
        for tokens_in_sentence in text_tokens_in_sentences:
            tokens += tokens_in_sentence
    max_text_tokens_len = max(max_text_tokens_len, len(tokens))
    X.append(tokens)
X = pad_and_encode(X, w2ind, max_text_tokens_len)
len(X)
X_train = [X[ind] for ind in inds_train]
X_val = [X[ind] for ind in inds_val]
n_samples_train = len(X_train)
n_samples_val = len(X_val)
class TrainValDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)
batch_size_train = 64
batch_size_val = 64

train_loader = DataLoader(TrainValDataset(X_train, y_train), batch_size=batch_size_train,
                          shuffle=False, num_workers=8, pin_memory=if_use_gpu)
val_loader = DataLoader(TrainValDataset(X_val, y_val), batch_size=batch_size_val,
                        shuffle=False, num_workers=8, pin_memory=if_use_gpu)
n_batches_train = len(train_loader)
class CNN_NLP(nn.Module):

    def __init__(self, pretrained_embeddings=None, freeze_embedding=False, vocab_size=None,
                 embed_dim=300, filter_sizes=[3, 4, 5], n_filters=[100, 100, 100],
                 n_classes=5, dropout=0.5):
        """
        Args:
            pretrained_embeddings (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretrained
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            n_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 5
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()

        if pretrained_embeddings is not None:
            self.vocab_size, self.embed_dim = pretrained_embeddings.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                          padding_idx=0, max_norm=5.0)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=n_filters[i], kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(n_filters), n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """
        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size, n_classes)
        """

        x_embed = self.embedding(input_ids).float()

        x_reshaped = x_embed.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc(self.dropout(x_fc))

        return logits
freeze_embedding = False
filter_sizes = [3, 4, 5]
n_filters = [100, 100, 100]

model = CNN_NLP(pretrained_embeddings=pretrained_embeddings, freeze_embedding=freeze_embedding,
                embed_dim=emb_dim, filter_sizes=filter_sizes, n_filters=n_filters, n_classes=n_classes)
print(model)
if if_use_gpu:
    model.cuda()
lr = 1e-3
n_epochs = 15

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, threshold=0.1, verbose=True)
def train(epoch, if_use_gpu, model, train_loader, criterion, optimizer, scheduler,
          n_samples_train, batch_size_train, log_batches_interval=50):
    model.train()

    running_loss = 0.
    correct = 0

    for batch_ind, (X_train_batch, y_train_batch) in enumerate(train_loader):
        if if_use_gpu:
            X_train_batch = X_train_batch.cuda()
            y_train_batch = y_train_batch.cuda()

        output = model(X_train_batch)
        loss = criterion(output, y_train_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

        y_train_pred_batch = output.max(dim = 1)[1]
        correct += (y_train_pred_batch == y_train_batch.data).sum()

        if batch_ind % log_batches_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_ind * batch_size_train, n_samples_train,
                100. * batch_ind / n_batches_train, loss.data.item()
            ))

    print('\nTraining set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Lr: {}\n'.format(
        running_loss / n_samples_train, correct, n_samples_train,
        100. * correct / n_samples_train, scheduler.optimizer.param_groups[0]['lr']
    ))

def validation(if_use_gpu, model, val_loader, criterion, scheduler, n_samples_val):
    model.eval()

    running_loss = 0.
    correct = 0

    for X_val_batch, y_val_batch in val_loader:
        with torch.no_grad():
            if if_use_gpu:
                X_val_batch = X_val_batch.cuda()
                y_val_batch = y_val_batch.cuda()

            output = model(X_val_batch)
            loss = criterion(output, y_val_batch)

            running_loss += loss.data.item()

            y_val_pred_batch = output.max(dim = 1)[1]
            correct += (y_val_pred_batch == y_val_batch.data).cpu().sum()

    running_loss /= n_samples_val

    scheduler.step(np.around(running_loss, 2))

    print('\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_loss, correct, n_samples_val,
        100. * correct / n_samples_val
    ))
%%time
for epoch in range(1, n_epochs + 1):
    train(epoch, if_use_gpu, model, train_loader, criterion, optimizer,
          scheduler, n_samples_train, batch_size_train)

    validation(if_use_gpu, model, val_loader,
               criterion, scheduler, n_samples_val)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    torch.save(state, os.path.join(cnn_dir, 'model_' + str(epoch) + '.pth'))

    print('-' * 20)

print('Finished training!')
checkpoint = torch.load(os.path.join(cnn_dir, 'model_1.pth'))
checkpoint.keys()
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
# на каких данных обучаться [0 - тексты, 2 - заголовки, 1 - и то и другое]
flag_data_train = 1
with open(os.path.join(preprocessed_dir, 'texts.pkl'), 'rb') as infile:
    texts = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'titles.pkl'), 'rb') as infile:
    titles = pickle.load(infile)

with open(os.path.join(preprocessed_dir, 'test_texts.pkl'), 'rb') as infile:
    test_texts = pickle.load(infile)
with open(os.path.join(preprocessed_dir, 'test_titles.pkl'), 'rb') as infile:
    test_titles = pickle.load(infile)

test_data = pd.read_csv(os.path.join(data_dir, 'lenta-ru-test.csv'))
if flag_data_train == 0:
    X = texts
    X_test = test_texts
elif flag_data_train == 2:
    X = titles
    X_test = test_titles
else:
    X = list(map(' '.join, zip(titles, texts)))
    X_test = list(map(' '.join, zip(test_titles, test_texts)))
len(X), len(X_test)
with open(os.path.join(best_dir, 'data.txt'), 'w+') as outfile:
    for i in range(n_samples):
        outfile.write(f'__label__{y[i]} {X[i]}\n')
%time best_classifier = fasttext.train_supervised(os.path.join(best_dir, 'data.txt'))
# где будет лежать обученный классификатор
best_classifier_file_name = 'classifier.bin'
# best_classifier.save_model(os.path.join(best_dir, best_classifier_file_name))
best_classifier = fasttext.load_model(os.path.join(best_dir, best_classifier_file_name))
y_test_pred = best_classifier.predict(X_test)[0]
y_test_pred = [label[0].split('_')[-1] for label in y_test_pred]
y_test_pred = np.array(y_test_pred, dtype=int)
submission = pd.DataFrame({'id': test_data.index,
                           'category': y_test_pred})
submission.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)
