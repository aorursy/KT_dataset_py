! pip install pymorphy2[fast] rnnmorph pymystem3 nltk razdel
text = 'купил таблетки от тупости,,, но не смог открыть банку,ЧТО ДЕЛАТЬ???'
text = text.lower()
text
import re
regex = re.compile(r'(\W)\1+')
regex.sub(r'\1', text)
regex = re.compile(r'[^\w\s]')
text=regex.sub(r' ', text).strip()
text
re.sub('\s+', ' ', text)
text = 'Купите кружку-термос "Hello Kitty" на 0.5л (64см³) за 3 рубля. До 01.01.2050.'
text.split()
from pymorphy2.tokenizers import simple_word_tokenize

simple_word_tokenize(text)
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize

sentences=sent_tokenize(text)
[word_tokenize(sentence) for sentence in sentences]
[wordpunct_tokenize(sentence) for sentence in sentences]
import razdel
sents=[]
for sentence in razdel.sentenize(text):
    sents.append(sentence.text)
sents
sentences = [sentence.text for sentence in razdel.sentenize(text)]

tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
tokens
razdel.tokenize
import razdel


def tokenize_with_razdel(text):
    sentences = [sentence.text for sentence in razdel.sentenize(text)]
    tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
    
    return tokens


tokenize_with_razdel(text)
from nltk.stem.snowball import SnowballStemmer

SnowballStemmer(language='english').stem('running')
SnowballStemmer(language='russian').stem('бежать')
from pymorphy2 import MorphAnalyzer

pymorphy = MorphAnalyzer()


def lemmatize_with_pymorphy(tokens):
    lemms = [pymorphy.parse(token)[0].normal_form for token in tokens]
    return lemms
tokens=['Купите',
  'кружку-термос',
  '"',
  'Hello',
  'Kitty',
  '"',
  'на',
  '0.5',
  'л',
  '(',
  '64',
  'см³',
  ')',
  'за',
  '3',
  'рубля',
  '.']
pymorphy.parse('бежал')
lemmatize_with_pymorphy(['бегут', 'бежал', 'бежите'])
lemmatize_with_pymorphy(['мама', 'мыла', 'раму'])
pymorphy.normal_forms('на заводе стали увидел виды стали')
lemmatize_with_pymorphy(['на', 'заводе', 'стали', 'увидел', 'виды', 'стали'])
pymorphy.parse('директора')
from pymystem3 import Mystem

mystem = Mystem()


def lemmatize_with_mystem(text):
    lemms=[token for token in mystem.lemmatize(text) if token!=' '][:-1]
    
    return  lemms
lemmatize_with_mystem('бегал бежал ')
[token for token in mystem.lemmatize('бежал бежал') if token!=' '][:-1]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

words = ['NLP', 'is', 'awesome']

label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(words)
corpus_encoded
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(corpus_encoded.reshape(-1, 1))
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
vectorizer.fit(corpus)
vectors = vectorizer.transform(corpus)
corpus = [
    'Девочка любит кота Ваську',
    'Тот кто любит, не знает кто любит его',
    'Кто кого любит?',
    'Васька любит девочка?',
]
vectors.todense()
vectorizer.vocabulary_
from sklearn.feature_extraction.text import TfidfVectorizer

idf_vectorizer=TfidfVectorizer()
vectors = idf_vectorizer.fit_transform(corpus)
vectors.todense()
idf_vectorizer.vocabulary_
import pandas as pd

train = pd.read_csv('../input/lecture-5-embeddings/train.csv')
train.shape
train.head()
train.label.value_counts(normalize=True)
test = pd.read_csv('../input/lecture-5-embeddings/test.csv')
test.shape
test.label.value_counts(normalize=True)
test.head()
%matplotlib inline

import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def evaluate_vectorizer(vectorizer):
    train_vectors = vectorizer.fit_transform(train['text'])
    test_vectors = vectorizer.transform(test['text'])
    
    clf = LinearSVC(random_state=42)
    
    clf.fit(train_vectors, train['label'])
    
    predictions = clf.predict(test_vectors)
    
    print(classification_report(test['label'], predictions))
    
    return predictions
evaluate_vectorizer(CountVectorizer(min_df=2));
evaluate_vectorizer(TfidfVectorizer(min_df=2));
def tokenize_with_razdel(text):
    tokens = [token.text for token in razdel.tokenize(text)]
    
    return tokens
evaluate_vectorizer(TfidfVectorizer(min_df=2, tokenizer=tokenize_with_razdel));
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
)

predictions=evaluate_vectorizer(tfidf_vectorizer)
from nltk.corpus import stopwords
stopwords = stopwords.words("russian")
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
    stop_words=stopwords
)

evaluate_vectorizer(tfidf_vectorizer)
tfidf_vectorizer = TfidfVectorizer(
    min_df=2, 
    tokenizer=lambda text: lemmatize_with_pymorphy(tokenize_with_razdel(text)),
    stop_words=stopwords,
    ngram_range=(1, 2)
)

evaluate_vectorizer(tfidf_vectorizer)
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_heatmap(true, predicted):
    classes = true.unique()
    matrix = confusion_matrix(true, predicted, labels=classes)
    sns.heatmap(matrix, xticklabels=classes, yticklabels=classes, annot=True, fmt='g')
plot_confusion_matrix_heatmap(test.label, predictions)
import io
import gzip
import pathlib
import urllib.request

WORD2VEC_PATH = pathlib.Path('word2vec.bin')

if not WORD2VEC_PATH.exists():
    url = 'https://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz'
    with urllib.request.urlopen(url) as connection:
        compressed = connection.read()
            
    decompressed = gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb').read()
    WORD2VEC_PATH.write_bytes(decompressed)
import numpy as np
from sklearn.base import TransformerMixin


class Word2VecVectorizer(TransformerMixin):
    def __init__(self, vectors):
        self.vectors = vectors
        self.zeros = np.zeros(self.vectors.vector_size)
        
    def _get_text_vector(self, text):
        token_vectors = []
        for token in tokenize_with_mystem_pos(text):
            try:
                token_vectors.append(self.vectors[token])
            except KeyError: # не нашли такой токен в словаре
                pass
                
        if not token_vectors:
            return self.zeros

        text_vector = np.sum(token_vectors, axis=0)
        return text_vector / np.linalg.norm(text_vector)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.array([self._get_text_vector(text) for text in X])
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
word2vec.most_similar(positive=['путин_S', 'вашингтон_S'], negative=['москва_S'])
word2vec_mystem = Mystem(entire_input=False)


def tokenize_with_mystem_pos(text):
    result = []
    
    for item in word2vec_mystem.analyze(text):
        if item['analysis']:
            lemma = item['analysis'][0]['lex']
            pos = re.split('[=,]', item['analysis'][0]['gr'])[0]
            token = f'{lemma}_{pos}'
        else:
            token = f'{item["text"]}_UNKN'
            
        result.append(token)

    return result
import numpy as np
from sklearn.base import TransformerMixin


class Word2VecVectorizer(TransformerMixin):
    def __init__(self, vectors):
        self.vectors = vectors
        self.zeros = np.zeros(self.vectors.vector_size)
        
    def _get_text_vector(self, text):
        token_vectors = []
        for token in tokenize_with_mystem_pos(text):
            try:
                token_vectors.append(self.vectors[token])
            except KeyError: # не нашли такой токен в словаре
                pass
                
        if not token_vectors:
            return self.zeros

        text_vector = np.sum(token_vectors, axis=0)
        return text_vector / np.linalg.norm(text_vector)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.array([self._get_text_vector(text) for text in X])
word2vec_vectorizer = Word2VecVectorizer(word2vec)

evaluate_vectorizer(word2vec_vectorizer);
from sklearn.pipeline import FeatureUnion

evaluate_vectorizer(
    FeatureUnion(
        [
            ('tf-idf', tfidf_vectorizer),
            ('word2vec', word2vec_vectorizer),
        ]
    )
);
def tokenize_with_razdel(text):
    return [token.text for token in razdel.tokenize(text)]
import razdel

train_texts = train['text'].apply(tokenize_with_razdel)
test_texts = test['text'].apply(tokenize_with_razdel)
from gensim.models import Word2Vec

model = Word2Vec(train_texts, 
                 size=32,     # embedding vector size
                 min_count=5,  # consider words that occured at least 5 times
                 window=5).wv  # define context as a 5-word window around the target word
class MyWord2Vec(Word2VecVectorizer):
    def _get_text_vector(self, text):
        token_vectors = []
        for token in tokenize_with_razdel(text):
            try:
                token_vectors.append(self.vectors[token])
            except KeyError: # не нашли такой токен в словаре
                pass
                
        if not token_vectors:
            return self.zeros

        text_vector = np.sum(token_vectors, axis=0)
        return text_vector / np.linalg.norm(text_vector)
word2vec_vectorizer = MyWord2Vec(word2vec)

evaluate_vectorizer(word2vec_vectorizer);