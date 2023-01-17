# import libraries

# 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.corpus import stopwords 

#from wordcloud import WordCloud

import re

from tqdm import tqdm

import random

#import regex

import gensim



#from sklearn.manifold import TSNE

from sklearn.metrics import f1_score



from sklearn.preprocessing import StandardScaler

import os

import nltk



import gc



import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import f1_score

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix, classification_report



from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences 

from keras.layers import Flatten



from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, concatenate

from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D



from keras.layers.normalization import BatchNormalization 

from keras.initializers import RandomNormal 

from keras import regularizers

from keras.callbacks import *

import keras







from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")

from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 
data = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

print ("Number of data points:", data.shape)
test_data = pd.read_csv('../input/quora-insincere-questions-classification/test.csv') 

print ("Number of data points:", test_data.shape)
#nltk.download('stopwords')

stopword = stopwords.words('english')
# Question Length

data['q_len'] = data['question_text'].str.len()

# Number of words in the question

data['q_words'] = data['question_text'].apply(lambda row: len(row.split(" ")))

# Number of Upper character in question

data['u_chars'] = data['question_text'].apply(lambda row: sum(1 for c in row if c.isupper()))

# Number of lower character in question

data['l_chars'] = data['question_text'].apply(lambda row: sum(1 for c in row if c.islower()))

# Number of stopwords in question

data['n_stopwords'] = data['question_text'].apply(lambda row: sum(1 for word in row.split(" ") if word in stopword))

# Number of capital words in question

data['n_cap_words'] = data['question_text'].apply(lambda row: sum(1 for word in row.split(" ") if word.isupper()))

# Number of different words in question

data['n_diff_words'] = data['question_text'].apply(lambda row: len(set(row.split(" "))))

# Averge Word length

data['avg_word_len'] = data['question_text'].apply(lambda row: sum(len(i) for i in row.split(" "))/len(row.split(" ")))

# Number of numerical Values in the text

data['n_numerical_words'] = data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if word.isnumeric()))







# Number of simpley

data["nb_simley"] = data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if re.findall(r"^(:\(|:\))+$", word)))

# Number of special symbols

data["nb_symbols"] = data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if re.findall(r"[@_!#$%^&*()<>?/\|}{~:]", word)))

# Number of Punctions

data["nb_punct"] = data['question_text'].apply(lambda row: sum(1 for c in row if (c=="'" or c==';'  or c=="/" or c=='.')))





print (data.shape)

#data.head()
# Question Length

test_data['q_len'] = test_data['question_text'].str.len()

# Number of words in the question

test_data['q_words'] = test_data['question_text'].apply(lambda row: len(row.split(" ")))

# Number of Upper character in question

test_data['u_chars'] = test_data['question_text'].apply(lambda row: sum(1 for c in row if c.isupper()))

# Number of lower character in question

test_data['l_chars'] = test_data['question_text'].apply(lambda row: sum(1 for c in row if c.islower()))

# Number of stopwords in question

test_data['n_stopwords'] = test_data['question_text'].apply(lambda row: sum(1 for word in row.split(" ") if word in stopword))

# Number of capital words in question

test_data['n_cap_words'] = test_data['question_text'].apply(lambda row: sum(1 for word in row.split(" ") if word.isupper()))

# Number of different words in question

test_data['n_diff_words'] = test_data['question_text'].apply(lambda row: len(set(row.split(" "))))

# Averge Word length

test_data['avg_word_len'] = test_data['question_text'].apply(lambda row: sum(len(i) for i in row.split(" "))/len(row.split(" ")))

# Number of numerical Values in the text

test_data['n_numerical_words'] = test_data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if word.isnumeric()))



# Number of simpley

test_data["nb_simley"] = test_data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if re.findall(r"^(:\(|:\))+$", word)))

# Number of special symbols

test_data["nb_symbols"] = test_data['question_text'].apply(lambda row: sum(1 for word in row.split(" ")if re.findall(r"[@_!#$%^&*()<>?/\|}{~:]", word)))

# Number of Punctions

test_data["nb_punct"] = test_data['question_text'].apply(lambda row: sum(1 for c in row if (c=="'" or c==';' or c=="/" or c=='.')))



print (test_data.shape)

#test_data.head()
# https://www.kaggle.com/kentaronakanishi/18th-place-solution

puncts = [

    ',', '.', '"', ':', ')', '(', '-', '!', '?','|', ';', "'", '$', '&',

    '/', '[', ']', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',

    '·', '_', '{', '}', '©', '^', '®', '`', '→', '°', '€', '™', '›',

    '♥', '←', '×', '§', '″', '′', 'Â', '█', 'à', '…', '“', '★', '”',

    '–', '●', 'â', '►', '−', '¢', '¬', '░', '¶', '↑', '±',  '▾',

    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '⊕', '▼',

    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

    'è', '¸', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',

    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',

    '¹', '≤', '‡', '₹', '´'

]
abbreviations = {

    "ain't": "is not",

    "aren't": "are not",

    "can't": "cannot",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'll": "he will",

    "he's": "he is",

    "how'd": "how did",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how is",

    "I'd": "I would",

    "I'd've": "I would have",

    "I'll": "I will",

    "I'll've": "I will have",

    "I'm": "I am",

    "I've": "I have",

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

    "so's": "so as",

    "this's": "this is",

    "that'd": "that would",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "there would",

    "there'd've": "there would have",

    "there's": "there is",

    "here's": "here is",

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

    "who'd": "who would",

    "who're": "who are",

    "'re": " are",

    "tryin'": "trying",

    "doesn'": "does not",

    'howdo': 'how do',

    'whatare': 'what are',

    'howcan': 'how can',

    'howmuch': 'how much',

    'howmany': 'how many',

    'whydo': 'why do',

    'doI': 'do I',

    'theBest': 'the best',

    'howdoes': 'how does',

}
#! pip install regex
import regex

spells = {

    'colour': 'color',

    'centre': 'center',

    'favourite': 'favorite',

    'travelling': 'traveling',

    'counselling': 'counseling',

    'theatre': 'theater',

    'cancelled': 'canceled',

    'labour': 'labor',

    'organisation': 'organization',

    'wwii': 'world war 2',

    'citicise': 'criticize',

    'youtu.be': 'youtube',

    'youtu ': 'youtube ',

    'qoura': 'quora',

    'sallary': 'salary',

    'Whta': 'what',

    'whta': 'what',

    'narcisist': 'narcissist',

    'mastrubation': 'masturbation',

    'mastrubate': 'masturbate',

    "mastrubating": 'masturbating',

    'pennis': 'penis',

    'Etherium': 'ethereum',

    'etherium': 'ethereum',

    'narcissit': 'narcissist',

    'bigdata': 'big data',

    '2k17': '2017',

    '2k18': '2018',

    '2k19': '2020',

    'qouta': 'quota',

    'exboyfriend': 'ex boyfriend',

    'exgirlfriend': 'ex girlfriend',

    'airhostess': 'air hostess',

    'whst': 'what',

    'watsapp': 'whatsapp',

    'demonitisation': 'demonetization',

    'demonitization': 'demonetization',

    'demonetisation': 'demonetization',

    'quorans': 'quora user',

    'quoran': 'quora user',

    'pokémon': 'pokemon',

    'bacteries': 'batteries', 

    'yr old': 'years old',

}



codes = ['\x7f', '\u200b', '\xa0', '\ufeff', '\u200e', '\u202a', '\u202c', '\u2060', '\uf0d8', '\ue019', '\uf02d', '\u200f', '\u2061', '\ue01b']





langs1 = r'[\p{Katakana}\p{Hiragana}\p{Han}]' # regex

langs2 = r'[ஆய்தஎழுத்துஆயுதஎழுத்துशुषछछशुषدوउसशुष북한내제តើបងប្អូនមានមធ្យបាយអ្វីខ្លះដើម្បីរកឃើញឯកសារអំពីប្រវត្តិស្ត្រនៃប្រាសាទអង្គរវट्टरौरआदસંઘરાજ્યपीतऊनअहএকটিবাড়িএকটিখামারএরঅধীনেপদেরবাছাইপরীক্ষাএরপ্রশ্নওউত্তরসহকোথায়পেতেপারিص、。Емелядуракلكلمقاممقال수능ί서로가를행복하게기乡국고등학교는몇시간업니《》싱관없어나이रचा키کپڤ」मिलगईकलेजेकोठंडकऋॠऌॡर]'

compiled_langs1 = regex.compile(langs1)

compiled_langs2 = re.compile(langs2)

def _clean_math(x, compiled_re):

    return compiled_re.sub(' <math> ', x)
def preprocess(x):

    

    x = str(x).lower()

    return x

    

def _clean_unicode(x):

    for u in codes:

        if u in x:

            x = x.replace(u, '')

    return x



def clean_math(df):

    math_puncts = 'θπα÷⁴≠β²¾∫≥⇒¬∠＝∑Φ√½¼'

    math_puncts_long = [r'\\frac', r'\[math\]', r'\[/math\]', r'\\lim']

    compiled_math = re.compile('(%s)' % '|'.join(math_puncts))

    compiled_math_long = re.compile('(%s)' % '|'.join(math_puncts_long))

    df['question_text'] = df['question_text'].apply(lambda x: _clean_math(x, compiled_math_long))

    df['question_text'] = df['question_text'].apply(lambda x: _clean_math(x, compiled_math))

    return df



def clean_abbreviation(df, abbreviations):

    compiled_abbreviation = re.compile('(%s)' % '|'.join(abbreviations.keys()))

    #print (compiled_abbreviation)

    def replace(match):

        return abbreviations[match.group(0)]

    df['question_text'] = df["question_text"].apply(

        lambda x: _clean_abreviation(x, compiled_abbreviation, replace)

    )

    return df



def _clean_abreviation(x, compiled_re, replace):

    return compiled_re.sub(replace, x)





def clean_spells(df, spells):

    compiled_spells = re.compile('(%s)' % '|'.join(spells.keys()))

    def replace(match):

        return spells[match.group(0)]

    df['question_text'] = df["question_text"].apply(

        lambda x: _clean_spells(x, compiled_spells, replace)

    )

    return df

    

def _clean_spells(x, compiled_re, replace):

    return compiled_re.sub(replace, x)



def _clean_language(x, compiled_re):

    return compiled_re.sub(' <lang> ', x)





def _clean_puncts(x, puncts):

    x = str(x)

    # added space around puncts after replace

    for punct in puncts:

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x





def _clean_space(x, compiled_re):

    return compiled_re.sub(" ", x)

 


data["question_text"] = data["question_text"].fillna("").apply(preprocess)

data["question_text"] = data["question_text"].fillna("").apply(_clean_unicode)

data = clean_math(data)

data = clean_abbreviation(data, abbreviations)

data = clean_spells(data, spells)

data['question_text'] = data['question_text'].apply(lambda x: _clean_language(x, compiled_langs1))

data['question_text'] = data['question_text'].apply(lambda x: _clean_language(x, compiled_langs2))

data['question_text'] = data['question_text'].apply(lambda x: _clean_puncts(x, puncts))

compiled_re = re.compile(r"\s+")

data['question_text'] = data["question_text"].apply(lambda x: _clean_space(x, compiled_re))

#data.to_csv('/content/gdrive/My Drive/CaseStudy/data_after_preprocessing7l.csv', index=False)


test_data["question_text"] = test_data["question_text"].fillna("").apply(preprocess)

test_data["question_text"] = test_data["question_text"].fillna("").apply(_clean_unicode)

test_data = clean_math(test_data)

test_data = clean_abbreviation(test_data, abbreviations)

test_data = clean_spells(test_data, spells)

test_data['question_text'] = test_data['question_text'].apply(lambda x: _clean_language(x, compiled_langs1))

test_data['question_text'] = test_data['question_text'].apply(lambda x: _clean_language(x, compiled_langs2))

test_data['question_text'] = test_data['question_text'].apply(lambda x: _clean_puncts(x, puncts))

compiled_re = re.compile(r"\s+")

test_data['question_text'] = test_data["question_text"].apply(lambda x: _clean_space(x, compiled_re))



del puncts, spells, codes, langs1, langs2, compiled_langs1, compiled_langs2; gc.collect()
test_data.columns
# https://stackoverflow.com/questions/38420847/apply-standardscaler-to-parts-of-a-data-set



col_names = [ 'q_len', 'q_words', 'u_chars', 'l_chars',

       'n_stopwords', 'n_cap_words', 'n_diff_words', 'avg_word_len',

       'n_numerical_words', 'nb_simley', 'nb_symbols', 'nb_punct']



data_features = data[col_names]

test_data_features = test_data[col_names]

scaler = StandardScaler().fit(data_features.values)

train_features = scaler.transform(data_features.values)

test_features = scaler.transform(test_data_features.values)



data[col_names] = train_features

test_data[col_names] = test_features

del data_features, test_data_features, train_features, col_names, test_features, scaler; gc.collect()
#max_features = 194200

X_train_question = data['question_text'] 

X_test_question = test_data['question_text']

tokenizer = Tokenizer() 

tokenizer.fit_on_texts(list(X_train_question)) 

vocab_size = len(tokenizer.word_index) + 1 

print (vocab_size) 

max_features = vocab_size -1

train_encoded_docs = tokenizer.texts_to_sequences(X_train_question) 

test_encoded_docs = tokenizer.texts_to_sequences(X_test_question) 

max_length = 50

train_padded_docs = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post') 

test_padded_docs = pad_sequences(test_encoded_docs, maxlen = max_length, padding='post') 

print(train_padded_docs[0])



del X_train_question, X_test_question, train_encoded_docs, test_encoded_docs; gc.collect()
train_padded_docs = pd.DataFrame(train_padded_docs)

test_padded_docs = pd.DataFrame(test_padded_docs)

train_data = pd.concat([train_padded_docs, data], axis=1)

#test_data = pd.concat([test_padded_docs, data], axis=1)

print (train_data.shape)

#print (test_data.head())
y = train_data.target

X = train_data.drop(['qid', 'target'], axis=1)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 40, stratify= y)

test_data = test_data.drop(['qid'], axis= 1)

print (X.shape)

print (y.shape)

print (test_data.shape)

#X_train.head()
train_padded_docs = X.iloc[:,[i for i in range(0,50)]] #[np.arange(0,65)] ]

#cv_padded_docs = X_test.iloc[:,[i for i in range(0,65)]] #[np.arange(0,65)] ]

print (train_padded_docs.shape)



#print (cv_padded_docs.shape)
print (len(X.columns))

X_train = X.iloc[:, [i for i in range(50, len(X.columns))]]

#X_cv = X_test.iloc[:, [i for i in range(65, len(X_test.columns))]]

X_train.shape
del X; gc.collect()
# https://stackoverflow.com/questions/19371860/python-open-file-from-zip-without-temporary-extracting-it

import zipfile

archive = zipfile.ZipFile('../input/quora-insincere-questions-classification/embeddings.zip', 'r')

#glove_file = archive.read('glove.840B.300d/glove.840B.300d.txt')
# https://www.kaggle.com/wowfattie/3rd-place

def P(word): 

    "Probability of `word`."

    # use inverse of rank as proxy

    # returns 0 if the word isn't in the dictionary

    return - WORDS.get(word, 0)



def correction(word): 

    "Most probable spelling correction for word."

    return max(candidates(word), key=P)

def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or [word])

def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)



def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)

def edits2(word): 

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def singlify(word):

    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])
from tqdm import tqdm



def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')



embeddings_index = {}

for o in tqdm(archive.open('glove.840B.300d/glove.840B.300d.txt', 'r')):    

    o = o.decode("utf-8")

    key , value =  get_coefs(*o.split(" "))

    embeddings_index.update({key:value})



words =  list(embeddings_index.keys())



w_rank = {}



for i,word in enumerate(words):

    w_rank[word] = i

WORDS = w_rank
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



unknown_vector = np.zeros((300,), dtype=np.float32) - 1.

print(unknown_vector[:5])

for key, i in word_index.items():

    word = key

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

    word = key.upper()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

    word = key.capitalize()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

    word = ps.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

    word = lc.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

    word = sb.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_1[i] = embedding_vector

        continue

 

    if i> 1:

        word = correction(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix_1[i] = embedding_vector

            continue

    

    embedding_matrix_1[i] = unknown_vector



del embeddings_index

del unknown_vector

gc.collect()
#embedding_matrix_1.shape
embeddings_index = {}

i = 0

for o in tqdm(archive.open('paragram_300_sl999/paragram_300_sl999.txt', 'r')):   

    try:

        o = o.decode("utf-8").strip()

        if len(o)>100:

            key , value =  get_coefs(*o.split(" "))

            if len(value) == 300:

                embeddings_index.update({key:value})

            else:

                i += 1

    except:

        continue

print (i)
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



unknown_vector = np.zeros((300,), dtype=np.float32) - 1.

print(unknown_vector[:5])

for key, i in word_index.items():

    word = key

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue

    word = key.upper()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue

    word = key.capitalize()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue

    word = ps.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue

    word = lc.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue

    word = sb.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_2[i] = embedding_vector

        continue



    

    if i> 1:

        word = correction(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix_2[i] = embedding_vector

            continue

    

    embedding_matrix_2[i] = unknown_vector



del embeddings_index

del unknown_vector

gc.collect()



del data

gc.collect()
embeddings_index = {}

i = 0

for o in tqdm(archive.open('wiki-news-300d-1M/wiki-news-300d-1M.vec', 'r')):   

    try:

        o = o.decode("utf-8").strip()

        if len(o)>100:

            key , value =  get_coefs(*o.split(" "))

            if len(value) == 300:

                embeddings_index.update({key:value})

            else:

                i += 1

    except:

        continue

print (i)
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



unknown_vector = np.zeros((300,), dtype=np.float32) - 1.

print(unknown_vector[:5])

for key, i in word_index.items():

    word = key

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue

    word = key.upper()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue

    word = key.capitalize()

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue

    word = ps.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue

    word = lc.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue

    word = sb.stem(key)

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix_3[i] = embedding_vector

        continue



    

    if i> 1:

        word = correction(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix_3[i] = embedding_vector

            continue

    

    embedding_matrix_3[i] = unknown_vector



del embeddings_index

del unknown_vector

gc.collect()



#embedding_matrix = np.concatenate((embedding_matrix_1 , embedding_matrix_2), axis=1)

embedding_matrix = np.concatenate((embedding_matrix_1 , embedding_matrix_2, embedding_matrix_3), axis=1)

del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3;

gc.collect()
#print (test_data.columns)

X_train_stat = X_train.drop(['question_text'], axis = 1)

test_data1 = test_data.drop(['question_text'], axis=1)

print (test_data1.shape)

print (X_train_stat.shape)

test_padded_docs.shape
submission_cv = pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv')

# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py

# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

    



def f1(y_true, y_pred):

    '''

    metric from here 

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    '''

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Function API in Keras https://machinelearningmastery.com/keras-functional-api-deep-learning/ 

from keras.layers.normalization import BatchNormalization 

# from keras.initializers import RandomNormal 

# from keras import regularizers 

# from keras.layers import Conv1D

# from keras.layers import Flatten
input1 = Input(shape=(50,)) 

#input1 = Input(shape=(100,), name = 'Input_sequence_Text')

x = Embedding(max_features, 300 * 3, weights=[embedding_matrix], input_length= 50,  trainable=False, name = 'Embedding')(input1)

x = SpatialDropout1D(0.4, name='SpatialDropout')(x)

x = Bidirectional(LSTM(256, return_sequences=True), name= 'BidirectionLSTM128')(x)

#bidirectionLSTM2 = Bidirectional(CuDNNGRU(128, return_sequences=True), name= 'BidirectionLSTM2')(bidirectionLSTM)

x = Conv1D(64, kernel_size= 1, name='1D_Convolution64')(x)

#maxpool = AVGM

max_pool = GlobalMaxPooling1D(name="GlobalMaxPool")(x) 

#flattan1 = Flatten(name= 'Flatten1')(max_pool)



input2 = Input(shape=(12,), name = 'input_stat_featues') 

embed2 = Embedding(22, 40)(input2)

conv2 = Conv1D(64, kernel_size= 3, activation='relu', kernel_regularizer= regularizers.l2(0.002), name='Conv1d')(embed2) 

flatten2 = Flatten(name='Flatten2')(conv2)



x = concatenate([max_pool, flatten2], name='Concatenate')

x = Dense(128, activation="relu", name='1Dense128')(x)

x = Dropout(0.1, name='Dropout2')(x)

x = BatchNormalization(name='BatchNormalization')(x)

#dense3 = Dense(64, activation="relu", name='2Dense64')(batchnormal)





x = Dense(1, activation="sigmoid")(x)

model1 = Model(inputs=[input1, input2], outputs=x)

#print (model3.summary())

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
clr = CyclicLR(base_lr=0.001, max_lr=0.002,

               step_size=300., mode='exp_range',

               gamma=0.99994)



model1.fit([train_padded_docs, X_train_stat], y, batch_size=512, epochs=4, callbacks = [clr,])

train_pred1 = model1.predict([train_padded_docs, X_train_stat], batch_size=512, verbose=1, callbacks = [clr, ]) 

test_pred1 = model1.predict([test_padded_docs, test_data1], batch_size=512, verbose=1, callbacks=[clr, ])
input1 = Input(shape=(50,)) 

#input1 = Input(shape=(100,), name = 'Input_sequence_Text')

embedding = Embedding(max_features, 300 * 3, weights=[embedding_matrix], input_length= 50,  trainable=False, name = 'Embedding')(input1)

dropout = SpatialDropout1D(0.4, name='SpatialDropout')(embedding)

bidirectionLSTM = Bidirectional(LSTM(128, return_sequences=True), name= 'BidirectionLSTM128')(dropout)

bidirectionLSTM2 = Bidirectional(LSTM(128, return_sequences=True), name= 'BidirectionLSTM2')(bidirectionLSTM)

#conv11 = Conv1D(64, kernel_size= 1, name='1D_Convolution64')(bidirectionLSTM)

#maxpool = AVGM

max_pool1 = GlobalMaxPooling1D(name="GlobalMaxPool")(bidirectionLSTM)

max_pool2 = GlobalMaxPooling1D(name="GlobalMaxPool2")(bidirectionLSTM2) 

#flattan1 = Flatten(name= 'Flatten1')(max_pool)

conc = Concatenate()([max_pool1, max_pool2])



input2 = Input(shape=(12,), name = 'input_stat_featues') 

embed2 = Embedding(12, 50)(input2)

conv2 = Conv1D(64, kernel_size= 3, activation='relu', kernel_regularizer= regularizers.l2(0.002), name='Conv1d')(embed2) 

flatten2 = Flatten(name='Flatten2')(conv2)



merge = concatenate([conc, flatten2], name='Concatenate')

dense64 = Dense(128, activation="relu", name='1Dense128')(merge)

dropout2 = Dropout(0.1, name='Dropout2')(dense64)

batchnormal = BatchNormalization(name='BatchNormalization')(dropout2)

#dense3 = Dense(64, activation="relu", name='2Dense64')(batchnormal)





final = Dense(1, activation="sigmoid")(batchnormal)

model1 = Model(inputs=[input1, input2], outputs=final)



model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])

model1.fit([train_padded_docs, X_train_stat], y, batch_size=512, epochs= 4, callbacks = [clr,])
train_pred2 = model1.predict([train_padded_docs, X_train_stat], batch_size=512, verbose=1, callbacks = [clr, ]) 

test_pred2 = model1.predict([test_padded_docs, test_data1], batch_size=512, verbose=1, callbacks=[clr, ])
input1 = Input(shape=(50,)) 

#input1 = Input(shape=(100,), name = 'Input_sequence_Text')

embedding = Embedding(max_features, 300 * 3, weights=[embedding_matrix], input_length= 50,  trainable=False, name = 'Embedding')(input1)

dropout = SpatialDropout1D(0.3, name='SpatialDropout')(embedding)

bidirectionLSTM = Bidirectional(LSTM(256, return_sequences=True), name= 'BidirectionLSTM128')(dropout)





x1 = Conv1D(100, activation='relu', kernel_size=1, 

                padding='same', kernel_initializer= keras.initializers.glorot_uniform(seed=110000))(bidirectionLSTM)

x2 = Conv1D(80, activation='relu', kernel_size=2, 

                padding='same', kernel_initializer= keras.initializers.glorot_uniform(seed=120000))(bidirectionLSTM)

x3 = Conv1D(30, activation='relu', kernel_size=3, 

                padding='same', kernel_initializer= keras.initializers.glorot_uniform(seed=130000))(bidirectionLSTM)

x4 = Conv1D(12, activation='relu', kernel_size=5, 

                padding='same', kernel_initializer= keras.initializers.glorot_uniform(seed=140000))(bidirectionLSTM)







x1 = GlobalMaxPooling1D()(x1)

x2 = GlobalMaxPooling1D()(x2)

x3 = GlobalMaxPooling1D()(x3)



x4 = GlobalMaxPooling1D()(x4)

c = concatenate([x1, x2, x3, x4])





#flattan1 = Flatten(name= 'Flatten1')(max_pool)



input2 = Input(shape=(12,), name = 'input_stat_featues') 

embed2 = Embedding(12, 40)(input2)

conv2 = Conv1D(64, kernel_size= 3, activation='relu', kernel_regularizer= regularizers.l2(0.002), name='Conv1d')(embed2) 

flatten2 = Flatten(name='Flatten2')(conv2)



merge = concatenate([c, flatten2], name='Concatenate')

x = Dense(128, activation="relu", name='1Dense128')(merge)

x = Dropout(0.1, name='Dropout2')(x)

x = BatchNormalization(name='BatchNormalization')(x)

#dense3 = Dense(64, activation="relu", name='2Dense64')(batchnormal)





final = Dense(1, activation="sigmoid")(x)

model1 = Model(inputs=[input1, input2], outputs=final)



model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
model1.fit([train_padded_docs, X_train_stat], 

          y, 

          batch_size=512, epochs=4, callbacks = [clr,],

          )
train_pred4 = model1.predict([train_padded_docs, X_train_stat], batch_size= 512, verbose=1, callbacks = [clr, ])

test_pred4 = model1.predict([test_padded_docs, test_data1], batch_size= 512, verbose=1, callbacks=[clr, ])
pred_val_y = 0.3 * train_pred1  +  0.3 * train_pred2 + 0.4 * train_pred4 

pred_test_y = 0.3 * test_pred1 + 0.3 * test_pred2 + 0.4 * test_pred4 



thresholds = []

for thresh in np.arange(0.1, 0.5, 0.01):

    thresh = np.round(thresh, 2)

    res = f1_score(y, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("F1 score at threshold {0} is {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]

print("Best threshold: ", best_thresh)
pred_test_y = (pred_test_y > best_thresh ).astype(int)
submission_cv['prediction'] = pred_test_y

print (submission_cv.head())
submission_cv.to_csv("submission.csv", index=False)