# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import nltk
import pandas as pd
import numpy as np
import operator 
import re
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing.text import Tokenizer
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
df = pd.concat([train ,test])

print("Number of texts: ", df.shape[0])
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
print("Extracting Paragram embedding")
embed_paragram = load_embed(paragram)
df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())
# build a contraction mapping dictionary 
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known
# clean contractions in text data before embedding 
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
df['treated_question'] = df['lowered_question'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi','\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '' }
def clean_special_chars(text, punct, mapping):
    
    ## use a map to replace unknown characters with known ones.
    for p in mapping:
        text = text.replace(p, mapping[p])
    ## make sure there are spaces between words and punctuation
    for p in punct:
        text = text.replace(p, f' {p} ')
        
    return text

df['treated_question'] = df['treated_question'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
# build a word mapping dictionary for frequent mispells 
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

df['treated_question'] = df['treated_question'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

import re
def clean_numbers(x):
    if re.search("[0-9]+", x) != None:
        x = re.sub('[0-9]+',' {} '.format(re.search('[0-9]+',x).group()),x) 
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
df["treated_question_num"] = df["treated_question"].progress_apply(lambda x: clean_numbers(x))

# modified from load_glove
def build_embedding_matrix_v2(word_index, embed):
    all_embs = np.stack(embed.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    empty_vector = np.random.normal(emb_mean, emb_std,(embed_size,))
    return np.vstack([empty_vector, all_embs])
embedding_matrix = build_embedding_matrix_v2(_, embed_paragram)
word_index = {word:i for i,word in enumerate(embed_paragram.keys(),1)}

def label_sentence(s, word_index):
    return [word_index.get(x,0) for x in s.split()]

train_embed = df[:train.shape[0]][['qid', 'target', 'treated_question_num']]
test_embed = df[train.shape[0]:][['qid', 'treated_question_num']]

from sklearn.model_selection import train_test_split
train, val = train_test_split(train_embed, test_size=0.1)
maxlen = 60
from keras.preprocessing.sequence import pad_sequences
y_train = train['target']
y_val = val['target']
%%time
train_X = train['treated_question_num'].apply(label_sentence, args=(word_index,))
train_X = pad_sequences(train_X, maxlen=maxlen)
%%time
val_X = val['treated_question_num'].apply(label_sentence, args=(word_index,))
val_X = pad_sequences(val_X, maxlen=maxlen)
%%time
sub_data = test_embed['treated_question_num'].apply(label_sentence, args=(word_index,))
sub_data = pad_sequences(sub_data, maxlen=maxlen)
import tensorflow as tf
from tensorflow import keras
import gc

del df, train_embed, test_embed
gc.collect()
# del df, train_embed, test_embed

# model = keras.Sequential()
# model.add(keras.layers.Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], input_length=maxlen, trainable=False))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# model.summary()
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
#                               min_delta=0,
#                               patience=1,
#                               verbose=0, mode='auto')

# history = model.fit(train_X,
#                 y_train,
#                 epochs=100,
#                 batch_size=1024,
#                 validation_data=(val_X, y_val),
#                 verbose=1,
#                 callbacks=[early_stop,])
# from sklearn import metrics
# y_pred = model.predict(val_X, batch_size=1024, verbose=1)
# best_thresh = 0
# best_score = 0
# for thresh in np.arange(0.1, 0.9, 0.01):
#     thresh = np.round(thresh, 2)
#     score = metrics.f1_score(y_val, (y_pred>thresh).astype(int))
#     if score>best_score:
#         best_score=score
#         best_thresh=thresh
#     print("F1 score at threshold {0} is {1}".format(thresh, score))
#del model
import gc
gc.collect()
tf.reset_default_graph()
# tf.keras.backend.clear_session()
model = keras.Sequential()
model.add(keras.layers.Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], input_length=maxlen, trainable=False))
# model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')
history = model.fit(train_X,
                y_train,
                epochs=100,
                batch_size=1024,
                validation_data=(val_X, y_val),
                verbose=1,
                callbacks=[early_stop,])

from sklearn import metrics
y_pred = model.predict(val_X, batch_size=1024, verbose=1)
best_thresh = 0
best_score = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(y_val, (y_pred>thresh).astype(int))
    if score>best_score:
        best_score=score
        best_thresh=thresh
    print("F1 score at threshold {0} is {1}".format(thresh, score))
pred_test_y = model.predict([sub_data], batch_size=1024, verbose=0)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = pred_test_y > best_thresh
sub.to_csv("submission.csv", index=False)
