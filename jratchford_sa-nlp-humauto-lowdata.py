# Imports

import pandas as pd

import numpy as np

import re

import nltk.data

import nltk

import os

from collections import OrderedDict

from subprocess import check_call

from shutil import copyfile

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

import mpld3

import seaborn as sns

from collections import Counter

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from tqdm import tqdm

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalAveragePooling1D,Merge,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

from nltk import word_tokenize

from keras.layers.merge import concatenate

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.sequence import pad_sequences

from keras import initializers

from keras import backend as K

from sklearn.linear_model import SGDClassifier as sgd

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from time import time
start = time()

# Read data

print('Extract...',round(time()-start,0))

train = "../input/train.csv" #change this to correct training csv

test = "../input/test.csv" #change this to correct test csv

X_train_ = pd.read_csv( train, header=0,delimiter="," )

X_train=X_train_.sample(frac=0.3, random_state=12345)

X_test = pd.read_csv( test, header=0,delimiter="," )



authors = ['EAP','MWS','HPL']

Y_train = LabelEncoder().fit_transform(X_train['author'])
# Clean data

def clean(X_train,X_test):

    X_train['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_train['text']]

    X_test['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_test['text']]

    return X_train,X_test

X_train,X_test = clean(X_train,X_test)

print('Leaning words...',round(time()-start,0))



auth_wds = {'EAP':0,'HPL':0,'MWS':0}



wd = {}

for i, row in X_train.iterrows():

    for a in row['words']:

        if len(a) > 1:

            try: 

                wd[a][row['author']] = wd[a][row['author']] + 1

                auth_wds[row['author']] = auth_wds[row['author']] + 1

            except:

                c_eap = 0

                c_hpl = 0

                c_mws = 0

                try: c_eap = wd[a]['EAP'] 

                except: pass

                try: c_hpl = wd[a]['HPL'] 

                except: pass

                try: c_mws = wd[a]['EAP'] 

                except: pass   

                wd[a] = {'EAP':c_eap,'HPL':c_hpl,'MWS':c_mws}

                wd[a][row['author']] = wd[a][row['author']] + 1

                auth_wds[row['author']] = auth_wds[row['author']] + 1

                

def remove_key(dictionary,key):

    r = dict(dictionary)

    del r[key]

    return r        



for key in list(wd.keys()):

    pass

    if wd[key]['EAP'] + wd[key]['HPL'] + wd[key]['MWS'] < 100: 

        wd = remove_key(wd,key)

        

c_eap = 0

c_hpl = 0

c_mws = 0    

for key in list(wd.keys()): 

    pass

    if not any([(wd[key]['EAP']/auth_wds['EAP'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2,

          (wd[key]['EAP']/auth_wds['EAP'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2,

          (wd[key]['HPL']/auth_wds['HPL'])/((wd[key]['EAP']+1)/auth_wds['EAP'])>2.2,

          (wd[key]['HPL']/auth_wds['HPL'])/((wd[key]['MWS']+1)/auth_wds['MWS'])>2.2,

          (wd[key]['MWS']/auth_wds['MWS'])/((wd[key]['EAP']+1)/auth_wds['EAP'])>2.2,

         ( wd[key]['MWS']/auth_wds['MWS'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2]):

        wd = remove_key(wd,key)



col_wds = {}

for key in list(wd.keys()): 

    col_wds[key]=0



rows = []

for words in X_train['words']:

    line_wds = dict(col_wds)

    for word in words:

        try: line_wds[word] = line_wds[word] + 1

        except: pass

    row = [line_wds[key] for key in list(line_wds.keys())]

    rows.append(row)

pd_df = pd.DataFrame(rows)

    

for column in pd_df: 

    pass

    X_train['wd_'+str(column)] = pd_df[column]



rows = []

for words in X_test['words']:

    line_wds = dict(col_wds)

    for word in words:

        try: line_wds[word] = line_wds[word] + 1

        except: pass

    row = [line_wds[key] for key in list(line_wds.keys())]

    rows.append(row)

pd_df = pd.DataFrame(rows)

    

for column in pd_df: 

    pass

    X_test['wd_'+str(column)] = pd_df[column]



auth_wds = {'EAP':0,'HPL':0,'MWS':0}



wd = {}

for i, row in X_train.iterrows():

    for a in row['words']:

        if len(a) > 1:

            try: 

                wd[a][row['author']] = wd[a][row['author']] + 1

                auth_wds[row['author']] = auth_wds[row['author']] + 1

            except:

                c_eap = 0

                c_hpl = 0

                c_mws = 0

                try: c_eap = wd[a]['EAP'] 

                except: pass

                try: c_hpl = wd[a]['HPL'] 

                except: pass

                try: c_mws = wd[a]['EAP'] 

                except: pass   

                wd[a] = {'EAP':c_eap,'HPL':c_hpl,'MWS':c_mws}

                wd[a][row['author']] = wd[a][row['author']] + 1

                auth_wds[row['author']] = auth_wds[row['author']] + 1

                

def remove_key(dictionary,key):

    r = dict(dictionary)

    del r[key]

    return r        



for key in list(wd.keys()):

    pass

    if wd[key]['EAP'] + wd[key]['HPL'] + wd[key]['MWS'] < 5: 

        wd = remove_key(wd,key)

        

   

e = auth_wds['EAP']

h = auth_wds['HPL']

m = auth_wds['MWS']

eap_wds = []

hpl_wds = []

mws_wds = []

for key in list(wd.keys()): 

    pass

    if (wd[key]['EAP']/e)>((wd[key]['HPL']/h)+(wd[key]['MWS'])/m):

        eap_wds.append(key)

    elif (wd[key]['HPL']/e)>((wd[key]['EAP']/e)+(wd[key]['MWS'])/m):

        hpl_wds.append(key)

    elif (wd[key]['MWS']/e)>((wd[key]['HPL']/h)+(wd[key]['EAP'])/e):

        mws_wds.append(key)

c_wd_rows_eap = []

c_wd_rows_hpl = []

c_wd_rows_mws = []

dup_wds = []

for row in X_train['words']:

    if len(row) == len(set(row)): dup_wds.append(0)

    else: dup_wds.append((len(row)-len(set(row)))/len(row)*10)

    for word in row:

        c_eap = 0

        c_hpl = 0

        c_mws = 0 

        if word in eap_wds: c_eap+=1

        elif word in hpl_wds: c_hpl+=1

        elif word in mws_wds: c_mws+=1

    c_wd_rows_eap.append(c_eap)

    c_wd_rows_hpl.append(c_hpl)

    c_wd_rows_mws.append(c_mws)

X_train['c_wd_eap'] = c_wd_rows_eap  

X_train['c_wd_hpl'] = c_wd_rows_hpl  

X_train['c_wd_mws'] = c_wd_rows_mws 

X_train['dup_wds'] = dup_wds

c_wd_rows_eap = []

c_wd_rows_hpl = []

c_wd_rows_mws = []

dup_wds = []

for row in X_test['words']:

    if len(row) == len(set(row)): dup_wds.append(0)

    else: dup_wds.append((len(row)-len(set(row)))/len(row)*10)

    for word in row:

        c_eap = 0

        c_hpl = 0

        c_mws = 0 

        if word in eap_wds: c_eap+=1

        elif word in hpl_wds: c_hpl+=1

        elif word in mws_wds: c_mws+=1

    c_wd_rows_eap.append(c_eap)

    c_wd_rows_hpl.append(c_hpl)

    c_wd_rows_mws.append(c_mws)

X_test['c_wd_eap'] = c_wd_rows_eap  

X_test['c_wd_hpl'] = c_wd_rows_hpl  

X_test['c_wd_mws'] = c_wd_rows_mws 

X_test['dup_wds'] = dup_wds



print('Characters...',round(time()-start,0))

all_char = set([i for i in str(X_train['text'])])

for char in all_char:

    X_train['punc_'+char] = [(sum([1  for nchar in sentence if nchar == char])/len(sentence)) for sentence in X_train['text']]

    X_test['punc_'+char] = [(sum([1  for nchar in sentence if nchar == char])/len(sentence)) for sentence in X_test['text']]
from gensim.parsing.preprocessing import STOPWORDS



# Feature Engineering

# Stop Words

print('Other columns...',round(time()-start,0))

_dist_train = [x for x in X_train['words']]

X_train['stop_word'] = [len([word for word in sentence if word in STOPWORDS])*100.0/len(sentence) for sentence in _dist_train]



_dist_test = [x for x in X_test['words']]

X_test['stop_word'] = [len([word for word in sentence if word in STOPWORDS])*100.0/len(sentence) for sentence in _dist_test]  



## Number of words in the text ##

X_train["num_words"] = X_train["text"].apply(lambda x: len(str(x).split()))

X_test["num_words"] = X_test["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

X_train["num_unique_words"] = X_train["text"].apply(lambda x: len(set(str(x).split())))

X_test["num_unique_words"] = X_test["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

X_train["num_chars"] = X_train["text"].apply(lambda x: len(str(x)))

X_test["num_chars"] = X_test["text"].apply(lambda x: len(str(x)))



## Average length of the words in the text ##

X_train["mean_word_len"] = X_train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

X_test["mean_word_len"] = X_test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



print('TFIDF...',round(time()-start,0))

### Fit transform the count vectorizer ###

tfidf_vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1,3))

tfidf_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())

train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())

test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())



# Feature Engineering

# count - words - nb

def countWords(X_train,X_test):

    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))

    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())

    train_count = count_vec.transform(X_train['text'].values.tolist())

    test_count = count_vec.transform(X_test['text'].values.tolist())

    return train_count,test_count

    

def runMNB(train_X, train_y, test_X, test_y, test_X2):

    model = naive_bayes.MultinomialNB()

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)

    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model



def do_count_MNB(X_train,X_test,Y_train):

    train_count,test_count=countWords(X_train,X_test)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros([X_train.shape[0], 3])

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

    for dev_index, val_index in kf.split(X_train):

        dev_X, val_X = train_count[dev_index], train_count[val_index]

        dev_y, val_y = Y_train[dev_index], Y_train[val_index]

        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index,:] = pred_val_y

        cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    print("Mean cv score : ", np.mean(cv_scores))

    pred_full_test = pred_full_test /5.

    return pred_train,pred_full_test



pred_train,pred_test = do_count_MNB(X_train,X_test,Y_train)

X_train["count_words_nb_eap"] = pred_train[:,0]

X_train["count_words_nb_hpl"] = pred_train[:,1]

X_train["count_words_nb_mws"] = pred_train[:,2]

X_test["count_words_nb_eap"] = pred_test[:,0]

X_test["count_words_nb_hpl"] = pred_test[:,1]

X_test["count_words_nb_mws"] = pred_test[:,2]





# Feature Engineering

# tfidf - chars - nb

def tfidfWords(X_train,X_test):

    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,5),analyzer='char')

    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())

    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())

    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())

    return train_tfidf,test_tfidf

    

def runMNB(train_X, train_y, test_X, test_y, test_X2):

    model = naive_bayes.MultinomialNB()

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)

    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model



def do(X_train,X_test,Y_train):

    train_tfidf,test_tfidf = tfidfWords(X_train,X_test)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros([X_train.shape[0], 3])

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=88)

    for dev_index, val_index in kf.split(X_train):

        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

        dev_y, val_y = Y_train[dev_index], Y_train[val_index]

        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index,:] = pred_val_y

        cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    print("Mean cv score : ", np.mean(cv_scores))

    pred_full_test = pred_full_test /5.

    return pred_train,pred_full_test

pred_train,pred_test = do(X_train,X_test,Y_train)





X_train["tfidf_chars_nb_eap"] = pred_train[:,0]

X_train["tfidf_chars_nb_hpl"] = pred_train[:,1]

X_train["tfidf_chars_nb_mws"] = pred_train[:,2]

X_test["tfidf_chars_nb_eap"] = pred_test[:,0]

X_test["tfidf_chars_nb_hpl"] = pred_test[:,1]

X_test["tfidf_chars_nb_mws"] = pred_test[:,2]

print('SpaCy...',round(time()-start,0))
# incorporate spacy vectors

import spacy

import en_core_web_lg

import string

nlp = en_core_web_lg.load()



#Clean text before feeding it to spaCy

punctuations = string.punctuation



# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation

def cleanup_text(docs):

    texts = []

    for doc in docs:

        doc = nlp(doc)

        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

        tokens = [tok for tok in tokens if tok not in STOPWORDS and tok not in punctuations]

        tokens = ' '.join(tokens)

        texts.append(tokens)

    return pd.Series(texts)
spacy_cleaned_train = cleanup_text(X_train['text'])

spacy_cleaned_test =  cleanup_text(X_test['text'])
print('Spacy Vectors...',round(time()-start,0))

train_vec = [doc.vector for doc in nlp.pipe(list(spacy_cleaned_train), batch_size=500, n_threads=4)]

train_vec = np.array(train_vec)



X_train[['spacy_vec_'+str(i) for i in range(300)]] = pd.DataFrame(train_vec.tolist())
test_vec = [doc.vector for doc in nlp.pipe(spacy_cleaned_train, batch_size=500, n_threads=4)]

test_vec = np.array(train_vec)



X_test[['spacy_vec_'+str(i) for i in range(300)]] = pd.DataFrame(test_vec.tolist())
print('Gensim Train...',round(time()-start,0))

import gensim

from gensim.models import doc2vec

LabeledSentence = gensim.models.doc2vec.LabeledSentence



#Gensim doc2vec

corpus_train = [z.split() for z in spacy_cleaned_train]

corpus_test = [z.split() for z in spacy_cleaned_test]

corpus_all = corpus_train + corpus_test





def labelizeReviews(reviews, label_type):

    labelized = []

    for i,v in enumerate(reviews):

        label = '%s_%s'%(label_type, i)

        labelized.append(LabeledSentence(v, [label]))

    return labelized







X_train_lab = labelizeReviews(corpus_train, 'Train')

X_test_lab = labelizeReviews(corpus_test, 'Test')

All_lab = labelizeReviews(corpus_all, 'All')





model = doc2vec.Doc2Vec(All_lab, min_count=1, window=10, size=300, workers=6)





def getVecs(model, corpus, size, vecs_type):

    vecs = np.zeros((len(corpus), size))

    for i in range(0, len(corpus)):

        vecs[i] = model.docvecs[i]

    return vecs



print('Gensim Vectors...',round(time()-start,0))

gen_train_vecs = getVecs(model, X_train_lab, 300, 'Train')

gen_test_vecs = getVecs(model, X_test_lab, 300, 'Train')





X_train[['d2v_vec_'+str(i) for i in range(300)]] = pd.DataFrame(gen_train_vecs.tolist())

X_test[['d2v_vec_'+str(i) for i in range(300)]] = pd.DataFrame(gen_test_vecs.tolist())
#SVD

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,5),analyzer='char')

full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())

train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())

test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())



n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))



X_train[['svd_char_'+str(i) for i in range(20)]] = pd.DataFrame(train_svd)

X_test[['svd_char_'+str(i) for i in range(20)]] = pd.DataFrame(test_svd)
splits = 4
# Using Neural Networks and Facebook's Fasttext

earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')



# NN

def doAddNN(X_train,X_test,pred_train,pred_test):

    X_train["nn_eap"] = pred_train[:,0]

    X_train["nn_hpl"] = pred_train[:,1]

    X_train["nn_mws"] = pred_train[:,2]

    X_test["nn_eap"] = pred_test[:,0]

    X_test["nn_hpl"] = pred_test[:,1]

    X_test["nn_mws"] = pred_test[:,2]

    return X_train,X_test



def initNN(nb_words_cnt,max_len):

    model = Sequential()

    model.add(Embedding(nb_words_cnt,32,input_length=max_len))

    model.add(Dropout(0.3))

    model.add(Conv1D(64,

                     5,

                     padding='valid',

                     activation='relu'))

    model.add(Dropout(0.3))

    model.add(MaxPooling1D())

    model.add(Flatten())

    model.add(Dense(800, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    return model



def doNN(X_train,X_test,Y_train):

    max_len = 70

    nb_words = 10000

    

    print('Processing text dataset')

    texts_1 = []

    for text in X_train['text']:

        texts_1.append(text)



    print('Found %s texts.' % len(texts_1))

    test_texts_1 = []

    for text in X_test['text']:

        test_texts_1.append(text)

    print('Found %s texts.' % len(test_texts_1))

    

    tokenizer = Tokenizer(num_words=nb_words)

    tokenizer.fit_on_texts(texts_1 + test_texts_1)

    sequences_1 = tokenizer.texts_to_sequences(texts_1)

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))



    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)



    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)

    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)

    del test_sequences_1

    del sequences_1

    nb_words_cnt = min(nb_words, len(word_index)) + 1



    # we need to binarize the labels for the neural net

    ytrain_enc = np_utils.to_categorical(Y_train)

    

    kf = model_selection.KFold(n_splits=splits, shuffle=True, random_state=2017)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros([xtrain_pad.shape[0], 3])

    for dev_index, val_index in kf.split(xtrain_pad):

        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]

        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]

        model = initNN(nb_words_cnt,max_len)

        model.fit(dev_X, y=dev_y, batch_size=32, epochs=4, verbose=1,

                  validation_data=(val_X, val_y),callbacks=[earlyStopping])

        pred_val_y = model.predict(val_X)

        pred_test_y = model.predict(xtest_pad)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index,:] = pred_val_y

    return doAddNN(X_train,X_test,pred_train,pred_full_test/splits)



# Fast Text



def doAddFastText(X_train,X_test,pred_train,pred_test):

    X_train["ff_eap"] = pred_train[:,0]

    X_train["ff_hpl"] = pred_train[:,1]

    X_train["ff_mws"] = pred_train[:,2]

    X_test["ff_eap"] = pred_test[:,0]

    X_test["ff_hpl"] = pred_test[:,1]

    X_test["ff_mws"] = pred_test[:,2]

    return X_train,X_test





def initFastText(embedding_dims,input_dim):

    model = Sequential()

    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(3, activation='softmax'))



    model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model



def preprocessFastText(text_docs):

    text_docs = text_docs.replace("' ", " ' ")

    signs = set(',.:;"?!')

    prods = set(text_docs) & signs

    if not prods:

        return text_docs



    for sign in prods:

        text_docs = text_docs.replace(sign, ' {} '.format(sign) )

    return text_docs



def create_docs(df, n_gram_max=2):

    def add_ngram(q, n_gram_max):

            ngrams = []

            for n in range(2, n_gram_max+1):

                for w_index in range(len(q)-n+1):

                    ngrams.append('--'.join(q[w_index:w_index+n]))

            return q + ngrams

    docs = []

    for doc in df.text:

        doc = preprocessFastText(doc).split()

        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    return docs



def doFastText(X_train,X_test,Y_train):

    min_count = 2



    docs = create_docs(X_train)

    tokenizer = Tokenizer(lower=False, filters='')

    tokenizer.fit_on_texts(docs)

    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])



    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')

    tokenizer.fit_on_texts(docs)

    docs = tokenizer.texts_to_sequences(docs)



    maxlen = 300



    docs = pad_sequences(sequences=docs, maxlen=maxlen)

    input_dim = np.max(docs) + 1

    embedding_dims = 20



    # we need to binarize the labels for the neural net

    ytrain_enc = np_utils.to_categorical(Y_train)



    docs_test = create_docs(X_test)

    docs_test = tokenizer.texts_to_sequences(docs_test)

    docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)

    xtrain_pad = docs 

    kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=2017)

    pred_full_test = 0

    pred_train = np.zeros([xtrain_pad.shape[0], 3])

    for dev_index, val_index in kf.split(xtrain_pad):

        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]

        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]

        model = initFastText(embedding_dims,input_dim)

        model.fit(dev_X, y=dev_y, batch_size=32, epochs=28, verbose=1,

                  validation_data=(val_X, val_y),callbacks=[earlyStopping])

        pred_val_y = model.predict(val_X)

        pred_test_y = model.predict(docs_test)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index,:] = pred_val_y

    return doAddFastText(X_train,X_test,pred_train,pred_full_test/3)

print('Other cool methods..',round(time()-start,0))

X_train,X_test = doFastText(X_train,X_test,Y_train)

X_train,X_test = doNN(X_train,X_test,Y_train)
# Final Model

# XGBoost

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.1

    param['max_depth'] = 3

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = child

    param['subsample'] = 0.8

    param['colsample_bytree'] = colsample

    param['seed'] = seed_val

    num_rounds = 2000



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30, verbose_eval=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)

    if test_X2 is not None:

        xgtest2 = xgb.DMatrix(test_X2)

        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)

    return pred_test_y, pred_test_y2, model



def do(X_train,X_test,Y_train):

    drop_columns=["id","text","words"]

    x_train = X_train.drop(drop_columns+['author'],axis=1)

    x_test = X_test.drop(drop_columns,axis=1)

    y_train = Y_train

    

    kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=2017)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros([x_train.shape[0], 3])

    for dev_index, val_index in kf.split(x_train):

        dev_X, val_X = x_train.loc[dev_index], x_train.loc[val_index]

        dev_y, val_y = y_train[dev_index], y_train[val_index]

        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, x_test, seed_val=0, colsample=0.7)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index,:] = pred_val_y

        cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    print("cv scores : ", cv_scores)

    return pred_full_test/4

result = do(X_train,X_test,Y_train)



result = pd.DataFrame(list(result),columns=['EAP','HPL','MWS'])

result['id'] = X_test['id']

result.to_csv('output_easier_process_version.csv',index=False)

print('Time to completion: ',round(time()-start,0))
len(result)
result.to_csv('test.csv')