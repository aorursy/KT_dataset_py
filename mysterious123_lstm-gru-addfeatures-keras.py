import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import os 

from collections import Counter
from nltk import word_tokenize

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
import keras

embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 5 # how many times to iterate over all samples
n_splits = 5 # Number of K-fold Splits
SEED = 10
debug =0
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        #ALLmight
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix 
    
            
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector    
    return embedding_matrix
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' {punct} ')
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)
def parallelize_apply(df,func,colname,num_process,newcolnames):
    # takes as input a df and a function for one of the columns in df
    pool =Pool(processes=num_process)
    arraydata = pool.map(func,tqdm(df[colname].values))
    pool.close()
    
    newdf = pd.DataFrame(arraydata,columns = newcolnames)
    df = pd.concat([df,newdf],axis=1)
    return df

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 4)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


# some fetures 
# ['','','india/n','','quora','','sex','','','country/countries','china','','','chinese','','']
def add_features(df):
    df['question_text'] = df['question_text'].apply(lambda x:str(x))
    df["lower_question_text"] = df["question_text"].apply(lambda x: x.lower())
    # df = parallelize_apply(df,sentiment,'question_text',4,['sentiment','subjectivity']) 
    # df['sentiment'] = df['question_text'].progress_apply(lambda x:sentiment(x))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] 
    return df

def load_and_prec():
    if debug:
        train_df = pd.read_csv("../input/train.csv")[:80000]
        test_df = pd.read_csv("../input/test.csv")[:20000]
    else:
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ###################### Add Features ###############################
    #  https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
    
    train = add_features(train_df)
    test = add_features(test_df)
    
#     train = parallelize_dataframe(train_df, add_features)
#     test = parallelize_dataframe(test_df, add_features)
    
    # lower
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values


    
    features = train[['num_unique_words','words_vs_unique', 'total_length', 'capitals', 'caps_vs_length','num_words']].fillna(0)
    test_features = test[['num_unique_words','words_vs_unique', 'total_length', 'capitals', 'caps_vs_length','num_words']].fillna(0)
       
    # doing PCA to reduce network run times
    ss = StandardScaler()
    pc = PCA(n_components=5)
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)
    print('features shape: ', features.shape)
    
    ###########################################################################

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
#     # Splitting to training and a final test set    
#     train_X, x_test_f, train_y, y_test_f = train_test_split(list(zip(train_X,features)), train_y, test_size=0.2, random_state=SEED)    
#     train_X, features = zip(*train_X)
#     x_test_f, features_t = zip(*x_test_f)    
    
    #shuffling the data
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]
    
    return train_X, test_X, train_y, features, test_features, tokenizer.word_index
#     return train_X, test_X, train_y, x_test_f,y_test_f,features, test_features, features_t, tokenizer.word_index
#     return train_X, test_X, train_y, tokenizer.word_index
start = time.time()
# fill up the missing values
# x_train, x_test, y_train, word_index = load_and_prec()
x_train, x_test, y_train, features, test_features, word_index = load_and_prec() 
# x_train, x_test, y_train, x_test_f,y_test_f,features, test_features,features_t, word_index = load_and_prec() 
print(time.time()-start)
if debug:
    paragram_embeddings = np.random.randn(120000,300)
    glove_embeddings = np.random.randn(120000,300)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
else:
    glove_embeddings = load_glove(word_index)
    embedding_matrix = glove_embeddings
#     paragram_embeddings = load_para(word_index)
#     embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)    
np.shape(embedding_matrix)
from keras.models import *
from keras.layers import *
from keras import backend as K 

feature_size = features.shape[1]
print('features_size: ', feature_size, 'train_x: ', x_train.shape, 'maxlen: ', maxlen)
def get_model(hidden_size, lin_size, embedding_matrix=embedding_matrix):
    sequence_input = Input(shape=(maxlen, ), dtype='int32')   
    embedding_layer = Embedding(input_dim=len(embedding_matrix),output_dim=embed_size, weights=[embedding_matrix], 
                                input_length=maxlen,trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    
    x = Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(hidden_size, return_sequences=True))(x)    

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    feature_input = Input(shape=(feature_size, )) 
#     conc = concatenate([avg_pool, max_pool, feature_input])
    hidden_trans = Lambda(lambda x: K.permute_dimensions(x, (1,0,2))[-1])
    hidden_outp = hidden_trans(x)
    print(hidden_outp.shape)
    conc = concatenate([hidden_outp, avg_pool, max_pool, feature_input], axis=1)
    outp = Dense(lin_size, activation="relu")(conc)
    outp = Dense(1, activation='sigmoid')(outp)

    model = Model(inputs=[sequence_input, feature_input], outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
splits = 5
kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)

batch_size = 512
hidden_size = 70
lin_size = 16

preds = []
preds_vals = []
y_vals = []
fold = 0
acc_scores = 0
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# GPU = [0, 1]
# for i in GPU:
#     with tf.device('/gpu:{0}'.format(i)):
# with tf.device('/gpu: 0'):
for train_idx, val_idx in kf.split(x_train, y_train):
    x_train_f = x_train[train_idx]
    f_train_f = features[train_idx]
    y_train_f = y_train[train_idx]
    x_val_f = x_train[val_idx]
    f_val_f = features[val_idx]
    y_val_f = y_train[val_idx]

    # Output batch loss every epoch
    model = get_model(hidden_size, lin_size)
    model.fit([x_train_f, f_train_f], y_train_f,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose = 1,
              validation_data=([x_val_f, f_val_f], y_val_f))
    preds_val = model.predict([x_val_f, f_val_f], batch_size=batch_size)
    preds_vals.append(preds_val)
    y_vals.append(y_val_f)

    preds_test = model.predict([x_test, test_features])
    preds.append(preds_test)

    fold+=1
#         acc_scores += accuracy_score(y_val_f, preds_val)
#         print('Fold {}, ACC = {}'.format(fold, accuracy_score(y_val_f, preds_val)))       
#     print("Cross Validation ACC = {}".format(acc_scores/splits))

# threshold rearch
best_thresh = 0
best_score = 0
for thresh in np.arange(0.1,0.501,0.01):
    thresh = np.round(thresh, 2)
    scores = 0.
    for i in range(len(preds_vals)):
        score = f1_score(y_vals[i], (preds_vals[i]>thresh).astype(int))
        scores += score
    print("F1 score at threshold {0} is {1}".format(thresh, scores/len(preds_vals)))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print ("F1 score at threshold {0} is the best: {1}".format(best_thresh, best_score))
preds = np.asarray(preds)
print(preds.shape)
y_test = np.mean(preds, axis=0)[:, 0]
print(y_test.shape)
y_test = (y_test > 0.33).astype(np.int)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_test})
submit_df.to_csv("submission_lstm_gru_addfeatures.csv", index=False)
import pandas as pd
submit_df = pd.read_csv('../input/submissions/submission_lstm_gru_addfeatures.csv')
