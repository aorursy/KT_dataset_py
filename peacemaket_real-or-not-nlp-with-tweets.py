!pip install pyspellchecker
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from gensim.models import KeyedVectors
import nltk

import re
import time
import gc
from string import punctuation
from pprint import pprint
from functools import reduce
from ast import literal_eval

import json
import scipy
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
# import spellchecker
from spellchecker import SpellChecker
from matplotlib import pyplot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

nlp = spacy.load('en')

%matplotlib inline
# Any results you write to the current directory are saved as output.
sample_subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

def make_submission(predictions, filename='./sumbission.csv'):
    pd.DataFrame({
        'id': sample_subm.id.values,
        'target': predictions
    }).to_csv(filename, index=False, header=True)
EMBEDDINGS_PATH = '../input/glovetwitter100d/glove.twitter.27B.100d.txt'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDINGS_PATH))
# embeddings.get('goooooaaaaal', None)
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
# train.head()
print(f'Number of samples: {train.shape[0]}')
print(f'Target_distribution: {train.target.value_counts(normalize=True).round(5).to_dict()}')
print('NaN ratio per column:')
for column in train.columns:
    print(f'\t{column}: {train[column].isna().sum()}/{train.shape[0]}')
for target in [0, 1]:
    print(f'5 texts for {target} target:')
    for text in train[train.target == target]['text'].head().to_list():
        print(f'\t{text}')

CONTRACTIONS = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
'&amp': 'and'
} #contractions_dict from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

HTTP_PATTERN = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
EMOJI_PATTERN = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
SPECWORDS_PATTERN = re.compile('((?:(?:[^a-z\s\#\@]+?[a-z]+[^a-z\s\#\@]*?)|(?:[^a-z\s\#\@]*?[a-z]+[^a-z\s\#\@]+))+)', flags=re.IGNORECASE)#|(?:.*[a-zA-Z]+.+)
CONTRACTIONS_PATTERN = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))
STOPWORDS = stopwords.words('english')

def remove_html(text):
    return HTTP_PATTERN.sub('',text)

def remove_emoji(txt):
    return EMOJI_PATTERN.sub('', txt)

def remove_punct(txt):
    table = str.maketrans('', '', punctuation)
    return txt.translate(table)

def remove_stopwords(txt):
    txt = txt.split()
    return ' '.join([word for word in txt if word not in STOPWORDS])

def contract_text(txt):
    txt = re.sub('([^\s])[\.\:,\(\)\?\!\^\;\=]+', '\g<1> ', txt.lower())

    def expand_contractions(s):
        def replace(match):
            return CONTRACTIONS[match.group(0)]
        return CONTRACTIONS_PATTERN.sub(replace, s)
    
    txt = re.sub('(?:w/|-\s|\=\>)', '', expand_contractions(txt).replace('&', ' and '))
    txt = re.sub('(\s+)', ' ', txt)
    
    return txt.strip()

def extract_hashtag(txt):
    return ' '.join([x.strip() for x in re.findall(r'\#(.*?)(?:\s|$)', txt)])

def extract_citations(txt):
    return [x.strip() for x in re.findall(r'\'(.*?)\'', txt)]


def lemmatize(txt):
    return ' '.join([token.lemma_ for token in nlp(txt)]).strip()

def lemmatize_hashtags(lst):
    return [token.lemma_ for txt in lst for token in nlp(txt)]

def lemmatize_citations(lst):
    return [' '.join([token.lemma_ for token in nlp(txt)]).strip() for txt in lst]


def drop_hashtags(txt):
    txt = re.sub(r'\#.*?(?:\s|$)', '', txt).strip()
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def drop_at_words(txt):
    txt = re.sub(r'\@.*?(?:\s|$)', '', txt).strip()
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def drop_citations(txt):
    txt = re.sub(r'\'.*?\'', '', txt).strip()
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def word_count(txt):
    return len(txt.split())


spell = SpellChecker()
def unknown_words_fraction(text):
    uknown_words = 0
    txt_split = text.split()
    misspelled_words = spell.unknown(txt_split)
    for word in txt_split:
        if word in misspelled_words:
            uknown_words += 1
    return uknown_words / len(txt_split) if len(txt_split) > 1 else 0
class BaseFeatureGenerator:
    
    def __init__(self):
        self.isTrain = True
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['contain_http'] = X['text'].str.contains('http')

        X['text'] = X.text.apply(lambda x: remove_html(x))
        X['text'] = X.text.apply(lambda x: remove_emoji(x))

        if self.isTrain:
            print(f'Before droping duplicates: {X.shape[0]}')
            X.drop_duplicates('text', inplace=True)
            print(f'After droping duplicates: {X.shape[0]}')

        X['contain_at'] = X['text'].str.contains('@')
        X['contain_hashtag'] = X['text'].str.contains('#')
        X['at_count'] = X.text.apply(lambda x: x.count('@'))
        #drop @ words
        X.loc[X['at_count']>0, 'text'] = X.loc[X['at_count']>0, 'text'].apply(lambda x: drop_at_words(x))

        X['text_len'] = X.text.str.len()

        X['clean_text'] = X.text.apply(lambda x: contract_text(x))

        #extract hashtags
        X['hashtag'] = X.clean_text.apply(lambda x: extract_hashtag(x))
        X['hashtag_count'] = X.text.apply(lambda x: x.count('#'))
        #extract citations
        X['citations'] = X.clean_text.apply(lambda x: extract_citations(x))
        X['citations_count'] = X.citations.apply(len)

        #drop hashtags
        X['text_no_hashtags'] = X.clean_text.values
        X.loc[X['hashtag_count']>0, 'text_no_hashtags'] = X.loc[X['hashtag_count']>0, 'text_no_hashtags'].apply(lambda x: drop_hashtags(x))

        #drop citations
        X['text_no_citations'] = X.clean_text.values
        X.loc[X['citations_count']>0, 'text_no_citations'] = X.loc[X['citations_count']>0, 'text_no_citations'].apply(lambda x: drop_citations(x))

        #drop citations and hashtags
        X['text_no_hashtags_no_citations'] = X.text_no_hashtags.values
        X.loc[X['citations_count']>0, 'text_no_hashtags_no_citations'] = X.loc[X['citations_count']>0, 'text_no_hashtags_no_citations'].apply(lambda x: drop_citations(x))
        return X

class SpecificFeatureGenerator:
    
    def __init__(self, remove_punctuation=True, is_remove_stopwords=True, is_lemmatize=True):
        self.remove_punctuation = remove_punctuation
        self.is_remove_stopwords = is_remove_stopwords
        self.is_lemmatize = is_lemmatize
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.remove_punctuation:
            for cl in ['clean_text', 'text_no_hashtags', 'text_no_citations', 'text_no_hashtags_no_citations', 'hashtag']:
                X[cl] = X[cl].apply(lambda x: remove_punct(x))
            X.loc[X['citations_count']>0, 'citations'] = X.loc[X['citations_count']>0, 'citations'].apply(lambda x: [remove_stopwords(y) for y in x])
    #         for cl in ['hashtag', 'citations']:
    #             X.loc[X['%s_count'%cl]>0, cl] = X.loc[X['%s_count'%cl]>0, cl].apply(lambda x: [remove_punct(y) for y in x])

        if self.is_remove_stopwords:
            for cl in ['clean_text', 'text_no_hashtags', 'text_no_citations', 'text_no_hashtags_no_citations', 'hashtag']:
                X[cl] = X[cl].apply(lambda x: remove_stopwords(x))
            X.loc[X['citations_count']>0, 'citations'] = X.loc[X['citations_count']>0, 'citations'].apply(lambda x: [remove_stopwords(y) for y in x])
    #         for cl in ['hashtag', 'citations']:
    #             X.loc[X['%s_count'%cl]>0, cl] = X.loc[X['%s_count'%cl]>0, cl].apply(lambda x: [remove_stopwords(y) for y in x])

        #word counting
        for cl in ['clean_text', 'text_no_hashtags', 'text_no_citations', 'text_no_hashtags_no_citations']:
            X[cl + '_word_count'] = X[cl].apply(lambda x: word_count(x))

        if self.is_lemmatize:
            for cl, func in [
                               ('clean_text', lemmatize), 
                               ('text_no_hashtags', lemmatize), 
                               ('text_no_citations', lemmatize), 
                               ('text_no_hashtags_no_citations', lemmatize),
                               ('hashtag', lemmatize),
                               ('citations', lemmatize_citations)
                            ]:
                print('lemmatize %s...'%cl)
                X['lemma_%s'%cl] = X[cl].apply(lambda x: func(x))
                if cl not in ['hashtag', 'citations']:
                    X['lemma_%s_unknown_words_fraction'%cl] = X['lemma_%s'%cl].apply(lambda x: unknown_words_fraction(x))

        else:
            for cl in ['clean_text', 'text_no_hashtags', 'text_no_citations', 'text_no_hashtags_no_citations']:
                X['lemma_%s_unknown_words_fraction'%cl] = X[cl].apply(lambda x: unknown_words_fraction(x))

        return X
    
    def set_params(self, remove_punctuation=True, is_remove_stopwords=True, is_lemmatize=True):
        self.remove_punctuation = remove_punctuation
        self.is_remove_stopwords = is_remove_stopwords
        self.is_lemmatize = is_lemmatize
def cross_validation(cv, model, X, y, metrics=[f1_score], verbose=True):
    
    scores = {}
    for metric in metrics:
        scores[metric.__name__] = {'train': [], 'val': []}
    
    for train_index, val_index in cv.split(X, y):
        X_train, X_val, y_train, y_val = X.loc[train_index], X.loc[val_index], y[train_index], y[val_index]

        model.fit(X_train, y_train)
        
        train_predictions_proba = model.predict_proba(X_train).T[1]
        val_predictions_proba = model.predict_proba(X_val).T[1]

        train_predictions = np.round(train_predictions_proba)
        val_predictions = np.round(val_predictions_proba)

        # metric calculation
        for index, metric in enumerate(metrics):
            if metric.__name__ in ['precision_recall_curve', 'roc_curve']:
                train_score = auc(*metric(y_train, train_predictions_proba)[:2][::-1])
                val_score = auc(*metric(y_val, val_predictions_proba)[:2][::-1])
            else:
                train_score = metric(y_train, train_predictions)
                val_score = metric(y_val, val_predictions)

            scores[metric.__name__]['train'].append(train_score)
            scores[metric.__name__]['val'].append(val_score)
            
    for metric in metrics:
        for key in ['train', 'val']:
            scores[metric.__name__][f'{key}_mean'] = np.mean(scores[metric.__name__][key]).round(5)
            scores[metric.__name__][key] = np.round(scores[metric.__name__][key], 5)
        if verbose:
            print(metric.__name__)
            print(f"Train: {scores[metric.__name__]['train']}, mean: {scores[metric.__name__]['train_mean']}")
            print(f"Val: {scores[metric.__name__]['val']}, mean: {scores[metric.__name__]['val_mean']}\n")
    
    return scores
    
train = pd.read_csv('../input/nlp-getting-started/train.csv')
tm = time.time()
feature_generator = Pipeline([
    ('baseFeatureGenerator', BaseFeatureGenerator()),
    ('specificFeatureGenerator', SpecificFeatureGenerator()),
])
feature_generator.fit(train, None)

train = feature_generator[0].transform(train)
for remove_punctuation in [True]:
    for is_remove_stopwords in [True]:
        for is_lemmatize in [True]:
# for remove_punctuation in [True, False]:
#     for is_remove_stopwords in [True, False]:
#         for is_lemmatize in [True, False]:
            print('remove_punctuation=%s, is_remove_stopwords=%s, is_lemmatize=%s'%(remove_punctuation, is_remove_stopwords, is_lemmatize))
            feature_generator[1].set_params(remove_punctuation, is_remove_stopwords, is_lemmatize)
            feature_generator[1].transform(train).to_csv('./train_%s_%s_%s.csv'%(remove_punctuation, is_remove_stopwords, is_lemmatize), header=True, index=False)
print(time.time() - tm)
# train = pd.read_csv('../input/nlp-getting-started/test.csv')
feature_generator[0].isTrain = False
feature_generator[1].set_params(True, True, True)

test = pd.read_csv('../input/nlp-getting-started/test.csv')
test = feature_generator.transform(test)
test.clean_text = test.clean_text.fillna('')
train = pd.read_csv('./train_%s_%s_%s.csv'%(True, True, True))
train.clean_text = train.clean_text.fillna('')
def twit2vec(df):
    vectors = []
    nums = 0
    for text in df['clean_text'].values:
        valid_tokens = [token for token in nltk.tokenize.word_tokenize(text) if token in embeddings]
#         valid_tokens = [token for token in text.split(' ') if token in embeddings]
        if valid_tokens:
            tokens_embeddings = [embeddings.get(token) for token in valid_tokens]
            vectors.append(np.mean(tokens_embeddings, axis=0))
        else:
            nums += 1
            vectors.append(np.zeros(100))
    print('number of empty twits:', nums)
    return vectors

# twit_vectors = twit2vec(train)
logreg = LogisticRegression(solver='lbfgs', max_iter=2000, C=1)

cv_params = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 123
}
cv = StratifiedKFold(**cv_params)


cross_validation(cv, logreg, pd.DataFrame(twit_vectors), train.target)
logreg = LogisticRegression(solver='lbfgs', max_iter=2000, C=1)
cv_params = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 123
}
cv = StratifiedKFold(**cv_params)

for remove_punctuation in [True, False]:
    for is_remove_stopwords in [True, False]:
        for is_lemmatize in [True, False]:
            print(remove_punctuation, is_remove_stopwords, is_lemmatize)
            train = pd.read_csv('./train_%s_%s_%s.csv'%(remove_punctuation, is_remove_stopwords, is_lemmatize))
            train.clean_text = train.clean_text.fillna('')
            twit_vectors = twit2vec(train)
            cross_validation(cv, logreg, pd.DataFrame(twit_vectors), train.target)
import torch
import torch.nn as nn
import torch.optim as optim
train = pd.read_csv('./train_%s_%s_%s.csv'%(True, True, True))
train.clean_text = train.clean_text.fillna('')
EMBEDDINGS_FILE = '../input/glovetwitter100d/glove.twitter.27B.100d.txt'
WORD2IDX = {}
WORD2VEC = {}
EMBEDDING_MATRIX = []
with open(EMBEDDINGS_FILE, 'r') as f:
    for idx, row in enumerate(f.readlines()):
        row = row.strip().split(' ')
        values = np.asarray(row[1:], dtype='float32')
        if values.shape[0] == 99:
            print(row[0], row)
            continue
#             break
        WORD2IDX[row[0]] = idx
        EMBEDDING_MATRIX.append(values)
        WORD2VEC[row[0]] = EMBEDDING_MATRIX[-1]
        
EMBEDDING_MATRIX = np.array(EMBEDDING_MATRIX)
def text2idx(texts):
    idx = []
    corpus = []
    for text in texts:
        idx.append([])
        corpus.append([])
        for word in text.split(' '):
            if word in WORD2IDX:
                idx[-1].append(WORD2IDX[word])
                corpus[-1].append(word)
    return idx, corpus

def groupby_token_count(idx, y):
    dic_X, dic_y = {}, {}
    for index, text in enumerate(idx):
        dic_X[len(text)] = dic_X.get(len(text), []) + [text]
        dic_y[len(text)] = dic_y.get(len(text), []) + [y[index]]
    return dic_X, dic_y

def train_val_split(grouped_X, grouped_y):
    grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y = {}, {}, {}, {}
    for token_count in grouped_X:
        if token_count == 0:
            continue
        indicator = False # whether minority class count is less than 2
        zero_cnt = one_count = 0
        for val in grouped_y[token_count]:
            if val == 1:
                one_count += 1
            else:
                zero_cnt += 1
            if one_count > 1 and zero_cnt > 1:
                break
        else:
            indicator = True
        if indicator:
            grouped_train_X[token_count] = grouped_X[token_count]
            grouped_train_y[token_count] = grouped_y[token_count]
        else:
            grouped_train_X[token_count], grouped_val_X[token_count], \
            grouped_train_y[token_count], grouped_val_y[token_count] = train_test_split(grouped_X[token_count], grouped_y[token_count], test_size=0.2, 
                                                                                        shuffle=True, random_state=123, stratify=grouped_y[token_count])
    return grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y
        

train_text2idx, train_corpus = text2idx(train.clean_text.values)
grouped_X, grouped_y = groupby_token_count(train_text2idx, train.target.values)
grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y = train_val_split(grouped_X, grouped_y)
print(grouped_X.keys())
def create_emb_layer(requires_grad=False):
    num_embeddings, embedding_dim = EMBEDDING_MATRIX.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.from_pretrained(torch.FloatTensor(EMBEDDING_MATRIX), freeze=~requires_grad)
#     print(emb_layer.weight.requires_grad)
#     emb_layer.weight.requires_grad = requires_grad
#     print(emb_layer.weight.requires_grad)

    return emb_layer, num_embeddings, embedding_dim

class RNNNet(nn.Module):
    def __init__(self, rnn_params={'hidden_size': 64, 'num_layers': 2, 'bidirectional': False, 'dropout': 0.2},):
        super().__init__()
        self.rnn_params = rnn_params
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(requires_grad=False)
        self.drop = nn.Dropout(0.2)
        self.gru = nn.GRU(embedding_dim, batch_first=True, **rnn_params)
        if rnn_params['bidirectional']:
            self.fc1 = nn.Linear(2*rnn_params['hidden_size'], 1)
        else:
            self.fc1 = nn.Linear(rnn_params['hidden_size'], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inp):
        x = self.embedding(inp)
        x = self.drop(x)
#         x = self.gru(x, torch.zeros((1, x.shape[0], self.rnn_params['hidden_size'],)))[0][:, -1, :]
        x = self.gru(x)[0][:, -1, :]
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    

net = RNNNet()
opt = optim.Adam(net.parameters(), lr=3e-4)#, lr=0.1)
criterion = nn.BCELoss()

# with torch.no_grad():
# #     inputs = prepare_sequences(train.clean_text.values[:1])#.reshape(1, 1, -1)
#     inputs = torch.tensor(grouped_X[7], dtype=torch.long)
#     tag_scores = net(inputs)
#     print(tag_scores)
def train_net(net, criterion, optimizer, train_X, val_X, train_y, val_y, n_epochs=10, batch_size=64):
    for epoch in range(n_epochs):
        net.train()
        train_loss = []
        train_target = []
#         for index in range(0, X.shape[0], batch_size):
        for token_count in train_X:
            for index in range(0, len(train_X[token_count]), batch_size):
                optimizer.zero_grad()
    #             if index % 100 == 0:
    #                 print(index, X.shape[0])
    #             train_X = prepare_sequences(X[index : index+batch_size])
    #             if train_X.shape[-1] == 0:
    #                 running_loss += [0]*batch_size
    #                 true_target += y[index : index+batch_size]
    #                 continue
    #             train_y = torch.tensor(y[index : index+batch_size], dtype=torch.float32).unsqueeze(0)
                if not token_count:
                    continue
                batch_X = torch.tensor(train_X[token_count][index : index+batch_size], dtype=torch.long)
                batch_y = torch.tensor(train_y[token_count][index : index+batch_size], dtype=torch.float32).unsqueeze(0)

                output = net(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += output.data.numpy().ravel().tolist()
                train_target += batch_y.data.numpy().ravel().astype(int).tolist()
    
        val_loss = []
        val_target = []
        net.eval()
        for token_count in val_X:
            for index in range(0, len(val_X[token_count]), batch_size):
                optimizer.zero_grad()
                if not token_count:
                    continue
                batch_X = torch.tensor(val_X[token_count][index : index+batch_size], dtype=torch.long)
                batch_y = torch.tensor(val_y[token_count][index : index+batch_size], dtype=torch.float32).unsqueeze(0)

                output = net(batch_X)
                val_loss += output.data.numpy().ravel().tolist()
                val_target += batch_y.data.numpy().ravel().astype(int).tolist()
        print('n_epoch: %s, train_f1_score:%s val_f1_score:%s train_accuracy:%s val_accuracy:%s'%(epoch, 
                                                                                                  round(f1_score(train_target, np.round(train_loss).astype(int), average='binary'), 5), 
                                                                                                  round(f1_score(val_target, np.round(val_loss).astype(int), average='binary'), 5),
                                                                                                 round(accuracy_score(train_target, np.round(train_loss).astype(int)), 5), 
                                                                                                 round(accuracy_score(val_target, np.round(val_loss).astype(int)), 5)))
                
    return net
    
net = train_net(net, criterion, opt, grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y, n_epochs=20, batch_size=64)

# del net
gc.collect()

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM, GRU, Dropout, Conv1D, Concatenate, GlobalAvgPool1D, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
train = pd.read_csv('./train_%s_%s_%s.csv'%(True, True, True))
train.clean_text = train.clean_text.fillna('')

EMBEDDINGS_FILE = '../input/glovetwitter100d/glove.twitter.27B.100d.txt'
WORD2IDX = {}
WORD2VEC = {}
EMBEDDING_MATRIX = []
with open(EMBEDDINGS_FILE, 'r') as f:
    for idx, row in enumerate(f.readlines()):
        row = row.strip().split(' ')
        values = np.asarray(row[1:], dtype='float32')
        if values.shape[0] == 99:
            print(row[0], row)
            continue
#             break
        WORD2IDX[row[0]] = idx
        EMBEDDING_MATRIX.append(values)
        WORD2VEC[row[0]] = EMBEDDING_MATRIX[-1]
        
EMBEDDING_MATRIX = np.array(EMBEDDING_MATRIX)
def text2idx(texts):
    idx = []
    corpus = []
    for text in texts:
        idx.append([])
        corpus.append([])
        for word in text.split(' '):
            if word in WORD2IDX:
                idx[-1].append(WORD2IDX[word])
                corpus[-1].append(word)
    return idx, corpus

train_text2idx, train_corpus = text2idx(train.clean_text.values)
test_text2idx, test_corpus = text2idx(test.clean_text.values)

def detirmineMaxLen(corpus):
    maxlen = 0
    for text in corpus:
        text_len = len(text)
        if text_len > maxlen:
            maxlen = text_len
    return maxlen

train_maxlen = detirmineMaxLen(train_corpus)
test_maxlen = detirmineMaxLen(test_corpus)
maxlen = max(train_maxlen, test_maxlen)
print(train_maxlen, test_maxlen)

# def groupby_token_count(idx, y):
#     dic_X, dic_y = {}, {}
#     for index, text in enumerate(idx):
#         dic_X[len(text)] = dic_X.get(len(text), []) + [text]
#         dic_y[len(text)] = dic_y.get(len(text), []) + [y[index]]
#     return dic_X, dic_y

# def train_val_split(grouped_X, grouped_y):
#     grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y = {}, {}, {}, {}
#     for token_count in grouped_X:
#         if token_count == 0:
#             continue
#         indicator = False # whether minority class count is less than 2
#         zero_cnt = one_count = 0
#         for val in grouped_y[token_count]:
#             if val == 1:
#                 one_count += 1
#             else:
#                 zero_cnt += 1
#             if one_count > 1 and zero_cnt > 1:
#                 break
#         else:
#             indicator = True
#         if indicator:
#             grouped_train_X[token_count] = grouped_X[token_count]
#             grouped_train_y[token_count] = grouped_y[token_count]
#         else:
#             grouped_train_X[token_count], grouped_val_X[token_count], \
#             grouped_train_y[token_count], grouped_val_y[token_count] = train_test_split(grouped_X[token_count], grouped_y[token_count], test_size=0.2, 
#                                                                                         shuffle=True, random_state=123, stratify=grouped_y[token_count])
#     return grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y
        

# grouped_X, grouped_y = groupby_token_count(train_text2idx, train.target.values)
# grouped_train_X, grouped_val_X, grouped_train_y, grouped_val_y = train_val_split(grouped_X, grouped_y)
# print(max(grouped_X.keys()))
train_padded_sequences = pad_sequences(train_text2idx, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.0)
test_padded_sequences = pad_sequences(test_text2idx, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.0)
train_padded_sequences = pad_sequences(train_text2idx, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
test_padded_sequences = pad_sequences(test_text2idx, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
class F1Score(keras.metrics.Metric):
    
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(**kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='falseneg', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(K.round(y_pred), tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.true_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.false_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, tf.float32)
        self.false_negatives.assign_add(tf.reduce_sum(values))
        
    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * recall * precision / (recall + precision + K.epsilon())
        return f1_score
    
model = Sequential()
model.add(Embedding(EMBEDDING_MATRIX.shape[0], EMBEDDING_MATRIX.shape[1],
                    embeddings_initializer=Constant(EMBEDDING_MATRIX), 
                    input_length=maxlen, trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=3e-4)

model.compile(optimizer=optimizer,
             loss='binary_crossentropy', 
             metrics=['accuracy', F1Score()])
# model.save_weights('model.h5')
# model.load_weights('model.h5')

train_X, val_X, train_y, val_y = train_test_split(train_padded_sequences, train.target.values, test_size=0.2, random_state=123, stratify=train.target.values)
history = model.fit(train_X, train_y, batch_size=64, epochs=10, validation_data=(val_X, val_y), verbose=2)#, callbacks=[MyCustomCallback()])
f1_score(train_y, np.round(model.predict(train_X)).ravel().astype(int), average='binary'), f1_score(val_y, np.round(model.predict(val_X)).ravel().astype(int), average='binary')
pd.Series(np.round(model.predict(test_padded_sequences)).ravel()).value_counts(normalize=True)
make_submission(np.round(model.predict(test_padded_sequences)).ravel().astype(int))
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM, GRU, Dropout, Conv1D, Concatenate, GlobalAvgPool1D, GlobalMaxPool1D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
train = pd.read_csv('./train_%s_%s_%s.csv'%(True, True, True))
train.clean_text = train.clean_text.fillna('')
test.clean_text = test.clean_text.fillna('')
EMBEDDINGS_FILE = '../input/glovetwitter100d/glove.twitter.27B.100d.txt'
WORD2IDX = {}
WORD2VEC = {}
EMBEDDING_MATRIX = []
with open(EMBEDDINGS_FILE, 'r') as f:
    for idx, row in enumerate(f.readlines()):
        row = row.strip().split(' ')
        values = np.asarray(row[1:], dtype='float32')
        if values.shape[0] == 99:
            print(row[0], row)
            continue
#             break
        WORD2IDX[row[0]] = idx
        EMBEDDING_MATRIX.append(values)
        WORD2VEC[row[0]] = EMBEDDING_MATRIX[-1]
        
EMBEDDING_MATRIX = np.array(EMBEDDING_MATRIX)
def text2idx(texts):
    idx = []
    corpus = []
    for text in texts:
        idx.append([])
        corpus.append([])
        for word in text.split(' '):
            if word in WORD2IDX:
                idx[-1].append(WORD2IDX[word])
                corpus[-1].append(word)
    return idx, corpus

train_text2idx, train_corpus = text2idx(train.clean_text.values)
test_text2idx, test_corpus = text2idx(test.clean_text.values)

def detirmineMaxLen(corpus):
    maxlen = 0
    for text in corpus:
        text_len = len(text)
        if text_len > maxlen:
            maxlen = text_len
    return maxlen

train_maxlen = detirmineMaxLen(train_corpus)
test_maxlen = detirmineMaxLen(test_corpus)
maxlen = max(train_maxlen, test_maxlen)
print(train_maxlen, test_maxlen)


train_padded_sequences = pad_sequences(train_text2idx, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.0)
test_padded_sequences = pad_sequences(test_text2idx, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.0)
train_padded_sequences = pad_sequences(train_text2idx, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
test_padded_sequences = pad_sequences(test_text2idx, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
class F1Score(keras.metrics.Metric):
    
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(**kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='falseneg', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(K.round(y_pred), tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.true_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.false_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, tf.float32)
        self.false_negatives.assign_add(tf.reduce_sum(values))
        
    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * recall * precision / (recall + precision + K.epsilon())
        return f1_score
    
embed_layer = Embedding(EMBEDDING_MATRIX.shape[0], EMBEDDING_MATRIX.shape[1],
                        embeddings_initializer=Constant(EMBEDDING_MATRIX), 
                        input_length=maxlen, trainable=False)
head1 = Sequential()
head1.add(embed_layer)
head1.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
head1.add(GlobalMaxPool1D())

head2 = Sequential()
head2.add(embed_layer)
head2.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
head2.add(GlobalMaxPool1D())

head3 = Sequential()
head3.add(embed_layer)
head3.add(Conv1D(filters=100, kernel_size=5, activation='relu'))
head3.add(GlobalMaxPool1D())

concat_layer = Concatenate([head1, head2, head3])

embed_layer = Embedding(EMBEDDING_MATRIX.shape[0], EMBEDDING_MATRIX.shape[1],
                        embeddings_initializer=Constant(EMBEDDING_MATRIX), 
                        input_length=maxlen, trainable=False)

sequence_input = Input(shape=(maxlen,), dtype='float32')
embedded_sequences = embed_layer(sequence_input)

head1 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedded_sequences)
head1 = GlobalMaxPool1D()(head1)

head2 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedded_sequences)
head2 = GlobalMaxPool1D()(head2)

head3 = Conv1D(filters=32, kernel_size=5, activation='relu')(embedded_sequences)
head3 = GlobalMaxPool1D()(head3)

concat_layer = concatenate([head1, head2, head3])

# fc1 = Dense(32, activation='relu')(concat_layer)
result = Dense(1, activation='sigmoid')(concat_layer)

model = Model(sequence_input, result)
optimizer = Adam(learning_rate=3e-4)

model.compile(optimizer=optimizer,
             loss='binary_crossentropy', 
             metrics=['accuracy', F1Score()])
model = Sequential()
model.add(embed_layer)
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=32, kernel_size=(5), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(GlobalMaxPool1D())
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=3e-4)

model.compile(optimizer=optimizer,
             loss='binary_crossentropy', 
             metrics=['accuracy', F1Score()])
# model.save_weights('model.h5')
# model.load_weights('model.h5')

train_X, val_X, train_y, val_y = train_test_split(train_padded_sequences, train.target.values, test_size=0.2, random_state=123, stratify=train.target.values)
history = model.fit(train_X, train_y, batch_size=32, epochs=15, validation_data=(val_X, val_y), verbose=2)#, callbacks=[MyCustomCallback()])
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM, GRU, Dropout, Conv1D, Concatenate, GlobalAvgPool1D, GlobalMaxPool1D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
import tensorflow_hub as hub

import tokenization
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
len(tf.config.experimental.list_physical_devices('GPU'))
def bert_encode(texts, tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
class F1Score(keras.metrics.Metric):
    
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(**kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='falseneg', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(K.round(y_pred), tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.true_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, tf.float32)
        self.false_positives.assign_add(tf.reduce_sum(values))
        
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, tf.float32)
        self.false_negatives.assign_add(tf.reduce_sum(values))
        
    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * recall * precision / (recall + precision + K.epsilon())
        return f1_score
    
def build_model(bert_layer, max_len=128):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy', F1Score()])
    
    return model
max_seq_length = 128  # Your choice here.
albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/1",
                              trainable=True)
# pooled_output, sequence_output = albert_layer([input_word_ids, input_mask, segment_ids])
sp_model_file = albert_layer.resolved_object.sp_model_file.asset_path.numpy()
tokenizer = tokenization.FullSentencePieceTokenizer(sp_model_file)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values
%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values
model = build_model(bert_layer, max_len=160)
# model.summary()
model = build_model(albert_layer, max_len=160)
# model.summary()
history = model.fit(train_input, train_labels, batch_size=16, epochs=3, validation_split=0.2, verbose=1)
len(tf.config.experimental.list_physical_devices('GPU'))