import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from textblob import TextBlob
from nltk import ngrams
from tqdm import tqdm
from collections import Counter

import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_colwidth = 170

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow_hub as hub
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# glove_embeddings = open('../input/glove6b100dtxt/glove.6B.100d.txt')
print(train.shape, test.shape)
train.target.value_counts()
plt.figure(figsize=(10, 5))
plt.pie(train.target.value_counts(),
            autopct='%1.2f%%',
            shadow=True,
            explode=(0.05, 0),
            startangle=60)
plt.legend(['Fake Diaster Tweet', 'Real Diaster Tweet'])
plt.title("Real and Fake Tweet Distribution", fontsize=16)
plt.show()
train = train.reindex(np.random.permutation(train.index))
train.head()
print(train.info(), "\n", test.info())
train_len = train.shape[0]
test_len = test.shape[0]

train_keyword_null_count = train[train.keyword.isnull() == True].shape[0]
test_keyword_null_count = test[test.keyword.isnull() == True].shape[0]

train_location_null_count = train[train.location.isnull() == True].shape[0]
test_location_null_count = test[test.location.isnull() == True].shape[0]

print("Training data having {}% Keyword as Null".format((train_keyword_null_count*100)/train_len))
print("Test data having {}% Keyword as Null\n".format((test_keyword_null_count*100)/test_len))

print("Training data having {}% Loaction as Null".format(train_location_null_count*100/train_len))
print("Test data having {}% Location as Null\n".format(test_location_null_count*100/test_len))
plt.figure(figsize=(10, 5))
plt.pie(train.keyword.isnull().value_counts(),
        autopct = '%1.2f%%',
        labels = ['Not Null', "Null"],
        shadow = True,
        explode=(0.05, 0),
        startangle=60)
plt.title('Keyword Distribution', fontsize=16)
plt.legend()
plt.show()
train.location.isnull().value_counts()
plt.figure(figsize=(10, 5))
plt.pie(train.location.isnull().value_counts(),
        autopct = '%1.2f%%',
        labels = ['Not Null', "Null"],
        shadow = True,
        explode=(0.05, 0),
        startangle=80)
plt.title('Location Distribution')
plt.legend()
plt.show()
train[~train.location.isnull()].location.head(10)
train.text[:10]
# Let's drop location column from train and test data.
train.drop("location", axis = 1, inplace = True)
test.drop("location", axis = 1, inplace = True)
# merge keyword with the text. You can keep it seperate also.
train.keyword.fillna("", inplace = True)
test.keyword.fillna("", inplace = True)

train.text = train.text + " " + train.keyword
test.text = test.text + " " + test.keyword
train.text[0:2]
# dropping keyword column
train.drop("keyword", axis = 1, inplace = True)
test.drop("keyword", axis = 1, inplace = True)
train_filter0 = train.target == 0
train_filter1 = train.target == 1
# Calculating length of texts
train["length"] = train.text.map(len)
test["length"] = test.text.map(len)
#lets check the histogram of length in train and test data
plt.figure(figsize=(12, 8))
sns.distplot(train.length, label = "Training Data", color='red')
sns.distplot(test.length, label = "Testing Data", color='yellow')
plt.title("Tweet Length Distribution(Train vs Test)", fontsize=16)
plt.legend(fontsize=12)
plt.show()
#lets check the histogram of length in training data for both targets
plt.figure(figsize=(12, 8))
sns.distplot(train[train_filter0].length, label = "Fake Diaster", color='red')
sns.distplot(train[train_filter1].length, label = "Real Diaster", color='yellow')
plt.title("Tweet Length Distribution(Training Data)", fontsize=16)
plt.legend(fontsize=12)
plt.show()
train["word_cnt"] = train.text.apply(lambda x : len(x.split(" ")))
test["word_cnt"] = test.text.apply(lambda x : len(x.split(" ")))

train["a_count"] = train.text.apply(lambda x : len([char for char in str(x) if char == "@"]))
test["a_count"] = test.text.apply(lambda x : len([char for char in str(x) if char == "@"]))

train["hash_count"] = train.text.apply(lambda x : len([char for char in str(x) if char == "#"]))
test["hash_count"] = test.text.apply(lambda x : len([char for char in str(x) if char == "#"]))
plt.figure(figsize=(12, 8))
sns.distplot(train[train_filter0].word_cnt, label = "Fake Diaster", color='red')
sns.distplot(train[train_filter1].word_cnt, label = "Real Diaster", color='yellow')
plt.title("Tweet Word Count Distribution(Training Data)", fontsize=16)
plt.legend(fontsize=12)
plt.show()
plt.figure(figsize=(12, 8))
sns.distplot(train[train_filter0].a_count, label = "Fake Diaster", color='red')
sns.distplot(train[train_filter1].a_count, label = "Real Diaster", color='green')
plt.title("Tweet @-tagging Count Distribution(Training Data)", fontsize=16)
plt.legend(fontsize=12)
plt.show()
plt.figure(figsize=(12, 8))
sns.distplot(train[train_filter0].hash_count, label = "Fake Diaster", color='red')
sns.distplot(train[train_filter1].hash_count, label = "Real Diaster", color='green')
plt.title("Tweet #-tag Count Distribution(Training Data)", fontsize=16)
plt.legend(fontsize=12)
plt.show()
def dict_formation(data):
    '''
    Module creates dictonary of words
    
    Input - Text
    
    Returns - Word Dictonary
    '''
    dict_word = {}
    #length = data.shape[0]
    for sent in data.text.tolist():
        words = sent.split(" ")
        for word in words:
            word = word.lower()
            try:
                dict_word[word] = dict_word[word]+1
            except:
                dict_word[word] = 1
    return dict_word
train_target0_words_dict = dict_formation(train[train_filter0])
train_target1_words_dict = dict_formation(train[train_filter1])

test_words_dict = dict_formation(test)
train_word_dict = dict_formation(train)
target0_words = train_target0_words_dict.keys()
target1_words = train_target1_words_dict.keys()

train_words = list(target0_words) + list(target1_words)

correlation_of_words_target01 = len(set(target0_words) & set(target1_words))*100/(len(target0_words) + len(target1_words))
correlation_of_words_train_test = len(set(train_words) & set(test_words_dict.keys()))*100/(len(train_words) + len(test_words_dict.keys()))

print("Train data target labels having common words {:.2f}% ".format(correlation_of_words_target01))
print("Train and Test data having common words {:.2f}% ".format(correlation_of_words_train_test))
def get_ngram_dataframe(n, data, label):
    train_ngram = ngrams(data.text.str.cat(sep=' ').split( ), n=n)
    train_ngram = Counter(train_ngram)
    train_ngram = dict(train_ngram)
    train_ngram = dict(sorted(train_ngram.items(), key=lambda x: x[1], reverse=True))

    train_ngram_df = pd.DataFrame()
    train_ngram_df[label] = train_ngram.keys()
    
    train_ngram_df['Count'] = train_ngram.values()
    
    return train_ngram_df
unigram_0 = get_ngram_dataframe(1, train[train_filter0], 'Unigram')
unigram_1 = get_ngram_dataframe(1, train[train_filter1], 'Unigram')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 12), constrained_layout=True)
fig.suptitle('Most common 30 Unigram', fontsize=16)
sns.barplot(y='Unigram', x='Count', data=unigram_0.head(30), color='red', ax=ax[0], label='Fake Diaster Tweet')
sns.barplot(y='Unigram', x='Count', data=unigram_1.head(30), color='green', ax=ax[1], label='Real Diaster Tweet')
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
plt.show()
bigram_0 = get_ngram_dataframe(2, train[train_filter0], 'Bigram')
bigram_1 = get_ngram_dataframe(2, train[train_filter1], 'Bigram')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 12), constrained_layout=True)
fig.suptitle('Most common 30 Bigrams', fontsize=16)
sns.barplot(y='Bigram', x='Count', data=bigram_0.head(30), color='red', ax=ax[0], label='Fake Diaster Tweet')
sns.barplot(y='Bigram', x='Count', data=bigram_1.head(30), color='green', ax=ax[1], label='Real Diaster Tweet')
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
plt.show()
trigram_0 = get_ngram_dataframe(3, train[train_filter0], 'Trigram')
trigram_1 = get_ngram_dataframe(3, train[train_filter1], 'Trigram')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 12), constrained_layout=True)
fig.suptitle('Most common 30 Trigram', fontsize=16)
sns.barplot(y='Trigram', x='Count', data=trigram_0.head(30), color='red', ax=ax[0], label='Fake Diaster Tweet')
sns.barplot(y='Trigram', x='Count', data=trigram_1.head(30), color='green', ax=ax[1], label='Real Diaster Tweet')
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
plt.show()
def load_embed(file):
    '''
    Module create the Glove embedding from the Glove text file.
    
    Input - Embedding file.
    
    Returns - Embedding Dictonary
    '''
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index
%%time
glove = '../input/glove6b100dtxt/glove.6B.100d.txt'
print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
def build_vocab(texts):
    '''
    Creates vocabulary
    
    Input - Text
    
    Returns - vocab Dictonary
    '''
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word.lower()] += 1
            except KeyError:
                vocab[word.lower()] = 1
    return vocab
import operator
def check_coverage(vocab, embeddings_index):
    '''
    To check coverage of the data vocabulary and embedding index
    
    Input:
        vocab - Data Vocabulary
        embeddings_index - Already trained embedding index
    
    Returns - Out of vocabulary dictionary and prints the coverge.
    '''
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    
    return unknown_words

# Lets check the embedding coverage before cleaning of data.
vocab_train = build_vocab(train['text'])
print("Glove : Train")
oov_glove_train = check_coverage(vocab_train, embed_glove)

vocab_test = build_vocab(test['text'])
print("Glove : Test")
oov_glove_test = check_coverage(vocab_test, embed_glove)
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                       "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", 
                       "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",
                       "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have" }
%%time
train.text = train.text.apply(lambda x: x.lower())
test.text = test.text.apply(lambda x: x.lower())
%%time
train.text = train.text.apply(lambda x : " ".join([contraction_mapping[word].lower() if word in contraction_mapping.keys() else word.lower() for word in x.split(" ")]))
test.text = test.text.apply(lambda x : " ".join([contraction_mapping[word].lower() if word in contraction_mapping.keys() else word.lower() for word in x.split(" ")]))
# Let's check coverage after contraction replacement. Yeyy, coverage percentage has increased.

vocab_train = build_vocab(train['text'])
print("Glove : Train")
oov_glove_train = check_coverage(vocab_train, embed_glove)

vocab_test = build_vocab(test['text'])
print("Glove : Test")
oov_glove_test = check_coverage(vocab_test, embed_glove)
def split_textnum(text):
    '''
    To seperate numbers from the words.
    
    Input - Word
    
    Returns - Number seperated list of items.
    '''
    match = re.match(r"([a-z]+)([0-9]+)", text, re.I)
    if match:
        items = " ".join(list(match.groups()))
    else:
        match = re.match(r"([0-9]+)([a-z]+)", text, re.I)
        if match:
            items = " ".join(list(match.groups()))
        else:
            return text
    return (items)
# Let's remove special charecters and unwanted datas. This is totally manual task, kind of real pain of NLP data cleaning.
def clean_text(text): 
            
    # Special characters
    text = re.sub(r"%20", " ", text)
    #text = text.replace(r".", " ")
    text = text.replace(r"@", " ")
    text = text.replace(r"#", " ")
    #text = text.replace(r":", " ")
    text = text.replace(r"'", " ")
    text = text.replace(r"\x89û_", " ")
    text = text.replace(r"??????", " ")
    text = text.replace(r"\x89ûò", " ")
    text = text.replace(r"16yr", "16 year")
    text = text.replace(r"re\x89û_", " ")
    
    text = text.replace(r"mh370", " ")
    text = text.replace(r"prebreak", "pre break")
    text = re.sub(r"\x89û", " ", text)
    text = re.sub(r"re\x89û", "re ", text)
    text = text.replace(r"nowplaying", "now playing")
    text = re.sub(r"\x89ûª", "'", text)
    text = re.sub(r"\x89û", " ", text)
    text = re.sub(r"\x89ûò", " ", text)
    
    
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏWhen", "When", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"China\x89Ûªs", "China's", text)
    text = re.sub(r"let\x89Ûªs", "let's", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"fromåÊwounds", "from wounds", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)    
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"SuruÌ¤", "Suruc", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"å£3million", "3 million", text)
    text = re.sub(r"åÀ", "", text)
    
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r"ªs", " ", text)
    text = re.sub(r"ª", " ", text)
    text = re.sub(r"\x9d", " ", text)
    text = re.sub(r"ò", " ", text)
    text = re.sub(r"ªt", " ", text)
    text = re.sub(r"ó", " ", text)
    text = text.replace(r"11yearold", "11 year old")
    text = re.sub(r"typhoondevastated", "typhoon devastated", text)
    text = re.sub(r"bestnaijamade", "best nijamade", text)
    text = re.sub(r"gbbo", "The Great British Bake Off", text)
    text = re.sub(r"ï", "", text)
    text = re.sub(r"ïwhen", "when", text)
    text = re.sub(r"selfimage", "self image", text)
    text = re.sub(r"20150805", "2015 08 05", text)
    text = re.sub(r"20150806", "2015 08 06", text)
    text = re.sub(r"subreddits", "website for weird public sentiment", text)
    text = re.sub(r"disea", "chinese famous electronic company", text)
    text = re.sub(r"lmao", "funny", text)
    text = text.replace(r"companyse", "company")
    
    text = text.replace(r"worldnews", "world news")
    text = text.replace(r"animalrescue", "animal rescue")
    text = text.replace(r"freakiest", "freak")
    
    text = text.replace(r"irandeal", "iran deal")
    text = text.replace(r"directioners", "mentor")
    text = text.replace(r"justinbieber", "justin bieber")
    text = text.replace(r"okwx", "okay")
    text = text.replace(r"trapmusic", "trap music")
    text = text.replace(r"djicemoon", "music ice moon")
    text = text.replace(r"icemoon", "ice moon")
    text = text.replace(r"mtvhottest", "tv hottest")
    text = text.replace(r"rì©union", "reunion")
    text = text.replace(r"abcnews", "abc news")
    text = text.replace(r"tubestrike", "tube strike")
    text = text.replace(r"prophetmuhammad", "prophet muhammad muslim dharma")
    text = text.replace(r"chicagoarea", "chicago area")
    text = text.replace(r"yearold", "year old")
    text = text.replace(r"meatloving", "meat love")
    text = text.replace(r"standuser", "standard user")
    text = text.replace(r"pantherattack", "panther attack")
    text = text.replace(r"youngheroesid", "young hearos id")
    text = text.replace(r"idk", "i do not know")
    text = text.replace(r"usagov", "united state of america government")
    text = text.replace(r"injuryi", "injury")
    text = text.replace(r"summerfate", "summer fate")
    text = text.replace(r"kerricktrial", "kerrick trial")
    text = text.replace(r"viralspell", "viral spell")
    text = text.replace(r"collisionno", "collision")
    text = text.replace(r"socialnews", "social news")
    text = text.replace(r"nasahurricane", "nasa hurricane")
    text = text.replace(r"strategicpatience", "strategic patience")
    text = text.replace(r"explosionproof", "explosion proof")
    text = text.replace(r"selfies", "photo")
    text = text.replace(r"selfie", "photo")
    text = text.replace(r"worstsummerjob", "worst summer job")
    text = text.replace(r"realdonaldtrump", "real america president")
    text = text.replace(r"omfg", "oh my god")
    text = text.replace(r"japìn", "japan")
    text = text.replace(r"breakingnews", "breaking news")
    
    text = " ".join([split_textnum(word) for word in text.split(" ")])
    
    text = "".join([c if c not in string.punctuation else "" for c in text])
    text = ''.join(c for c in text if not c.isdigit())
    text = text.replace(r"÷", "")
    
    text = re.sub(' +', ' ', text)
    # text = text.encode('utf-8')
    return text
%%time
train.text = train.text.apply(lambda x : clean_text(x))
test.text = test.text.apply(lambda x : clean_text(x))
train.text = train.text.apply(lambda x : " ".join([contraction_mapping[word].lower() if word in contraction_mapping.keys() else word.lower() for word in x.split(" ")]))
test.text = test.text.apply(lambda x : " ".join([contraction_mapping[word].lower() if word in contraction_mapping.keys() else word.lower() for word in x.split(" ")]))
# Let's check the coverage, Yeyy, It's improved again. We are moving in the right dirrection.

vocab_train = build_vocab(train['text'])
print("Glove : Train")
oov_glove_train = check_coverage(vocab_train, embed_glove)

vocab_test = build_vocab(test['text'])
print("Glove : Test")
oov_glove_test = check_coverage(vocab_test, embed_glove)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
train.text = train.text.apply(lambda x : "".join([lemmatizer.lemmatize(word) for word in x]))
test.text = test.text.apply(lambda x : "".join([lemmatizer.lemmatize(word) for word in x]))
vocab_train = build_vocab(train['text'])
print("Glove : Train")
oov_glove_train = check_coverage(vocab_train, embed_glove)

vocab_test = build_vocab(test['text'])
print("Glove : Test")
oov_glove_test = check_coverage(vocab_test, embed_glove)
import gc
del oov_glove_test
del embed_glove
gc.collect()
from sklearn.metrics import f1_score, recall_score, precision_score

class ClassificationReport(Callback):
    
    def __init__(self, train_data=(), validation_data=()):
        super(Callback, self).__init__()
        
        self.X_train, self.Y_train = train_data
        self.X_val, self.Y_val = validation_data
        
        self.train_precision_score = []
        self.train_recall_score = []
        self.train_f1_score = []
        
        self.val_precision_score = []
        self.val_recall_score = []
        self.val_f1_score = []
        
    def on_epoch_end(self, epoch, logs={}):
        
        train_prediction = np.round(self.model.predict(self.X_train, verbose=0))
        
        train_precision = precision_score(self.Y_train, train_prediction, average='macro')
        train_recall = recall_score(self.Y_train, train_prediction, average='macro')
        train_f1 = f1_score(self.Y_train, train_prediction, average='macro')
        
        self.train_precision_score.append(train_precision)
        self.train_recall_score.append(train_recall)
        self.train_f1_score.append(train_f1)
        
        val_prediction = np.round(self.model.predict(self.X_val, verbose=0))
        
        val_precision = precision_score(self.Y_val, val_prediction, average='macro')
        val_recall = recall_score(self.Y_val, val_prediction, average='macro')
        val_f1 = f1_score(self.Y_val, val_prediction, average='macro')
        
        self.val_precision_score.append(val_precision)
        self.val_recall_score.append(val_recall)
        self.val_f1_score.append(val_f1)
        
        print('\n Epoch - {} - Training Precision - {:.6} - Training Recall - {:.6} - Training F1-Score - {:.6}'.format(
        epoch+1, train_precision, train_recall, train_f1))
        
        print('\n Epoch - {} - Validation Precision - {:.6} - Validation Recall - {:.6} - Validation F1-Score - {:.6}'.format(
        epoch+1, val_precision, val_recall, val_f1))
        
# We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py


import tokenization
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)
class BertTraining:
    
    def __init__(self, bert_layer, fold_k=2, dropout=0.2, max_seq_len=160, lr=0.0001, epochs=15, batch_size=32):
        
        self.fold_k = fold_k
        self.bert_layer = bert_layer
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = []
        self.scores = {}
        
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        
    def bert_encode(self, texts):
        
        all_token_ids = []
        all_masks = []
        all_segments = []
        
        for text in texts:
            
            text = self.tokenizer.tokenize(text)
            text = text[: self.max_seq_len-2]
            input_seqence = ['[CLS]'] + text + ['[SEP]']
            padding_length = self.max_seq_len - len(input_seqence)
            
            tokens = self.tokenizer.convert_tokens_to_ids(input_seqence)
            tokens = tokens + [0]*padding_length
            pad_masks = [1]*len(input_seqence) + [0]*padding_length
            segment_ids = [0]*self.max_seq_len
            
            all_token_ids.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)
        
        return np.array(all_token_ids), np.array(all_masks), np.array(all_segments)
    
    def bert_model(self):
        
        input_token_id = Input(shape=(self.max_seq_len, ), dtype=tf.int32, name='input_token_id')
        input_mask = Input(shape=(self.max_seq_len, ), dtype=tf.int32, name='input_mask')
        input_segment = Input(shape=(self.max_seq_len, ), dtype=tf.int32, name='input_segment')
        
        _, sequence_output = self.bert_layer([input_token_id, input_mask, input_segment])
        clf_output = sequence_output[:, 0, :]
        
        if self.dropout == 0:
            output = Dense(1, activation='sigmoid')(clf_output)
        else:
            dropout = Dropout(self.dropout)(clf_output)
            output = Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=[input_token_id, input_mask, input_segment], outputs=output)
        
        optimizer = SGD(learning_rate=self.lr, momentum=0.8)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train_model(self, X):
        skf = StratifiedKFold(n_splits=self.fold_k, random_state=SEED, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X.text, X.text)):
            
            print("\n Fold {}\n".format(fold))
            
            X_Train_Encoded = self.bert_encode(X.loc[train_idx, 'text'].str.lower())
            Y_Train = X.loc[train_idx, 'target']
            
            X_Val_Encoded = self.bert_encode(X.loc[val_idx, 'text'].str.lower())
            Y_Val = X.loc[val_idx, 'target']
            
            metrics = ClassificationReport(train_data=(X_Train_Encoded, Y_Train), validation_data=(X_Val_Encoded, Y_Val))
            
            model = self.bert_model()
            model.fit(X_Train_Encoded, Y_Train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[metrics],
                     validation_data=(X_Val_Encoded, Y_Val), verbose=1)
            
            self.models.append(model)
            self.scores[fold] = {
                'train' : {
                    'precision' : metrics.train_precision_score,
                    'recall' : metrics.train_recall_score,
                    'f1_score': metrics.train_f1_score
                },
                
                'validation' : {
                    'precision' : metrics.val_precision_score,
                    'recall' : metrics.val_recall_score,
                    'f1_score': metrics.val_f1_score
                }
            }
            
    
    def plot_learning_curve(self):
        
        fig, axes = plt.subplots(nrows=self.fold_k, ncols=2, figsize=(20, self.fold_k * 6), dpi=100)
    
        for i in range(self.fold_k):
            
            # Classification Report curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[i].history.history['val_accuracy'], ax=axes[i][0], label='val_accuracy')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['precision'], ax=axes[i][0], label='val_precision')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['recall'], ax=axes[i][0], label='val_recall')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['f1_score'], ax=axes[i][0], label='val_f1')        

            axes[i][0].legend() 
            axes[i][0].set_title('Fold {} Validation Classification Report'.format(i), fontsize=14)

            # Loss curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['loss'], ax=axes[i][1], label='train_loss')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['val_loss'], ax=axes[i][1], label='val_loss')

            axes[i][1].legend() 
            axes[i][1].set_title('Fold {} Train / Validation Loss'.format(i), fontsize=14)

            for j in range(2):
                axes[i][j].set_xlabel('Epoch', size=12)
                axes[i][j].tick_params(axis='x', labelsize=12)
                axes[i][j].tick_params(axis='y', labelsize=12)

        plt.show()

    def predict(self, X_test):
        X_test_encode = self.bert_encode(X_test.text.str.lower())
        Y_pred = np.zeros((X_test_encode[0].shape[0], 1))
        
        for model in self.models:
            Y_pred += model.predict(X_test_encode)/len(self.models)
        
        return Y_pred
SEED = 42
clf = BertTraining(bert_layer, fold_k=3, dropout=0.5, max_seq_len=140, lr=0.0001, epochs=20, batch_size=64)

clf.train_model(train)
clf.plot_learning_curve()
prediction = clf.predict(test)
prediction
prediction = np.where(prediction < 0.5, 0, 1)
prediction
# test.head(2)
result = pd.DataFrame()
result["id"] = test['id']
result["target"] = np.squeeze(prediction)
result.head()
result.target.value_counts()
result.to_csv('submission.csv', index=False)
